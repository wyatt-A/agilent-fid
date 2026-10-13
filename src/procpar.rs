use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use indexmap::IndexMap;
use itertools::Itertools;
use regex::Regex;
use crate::AgilentFidError;

#[derive(Debug)]
enum BasicType {
    Undefined,
    Real,
    String,
}

impl BasicType {
    pub fn from_val(val: u8) -> Self {
        match val {
            0 => Self::Undefined,
            1 => Self::Real,
            2 => Self::String,
            _ => panic!("unexpected value"),
        }
    }
}

#[derive(Debug)]
enum SubType {
    Undefined,
    Real,
    String,
    Delay,
    Flag,
    Frequency,
    Pulse,
    Integer,
}

impl SubType {
    pub fn from_val(val: u8) -> Self {
        match val {
            0 => Self::Undefined,
            1 => Self::Real,
            2 => Self::String,
            3 => Self::Delay,
            4 => Self::Flag,
            5 => Self::Frequency,
            6 => Self::Pulse,
            7 => Self::Integer,
            _ => panic!("unexpected value"),
        }
    }
}

#[derive(Debug)]
struct ProcParField {
    pub sub_type: SubType,
    pub basic_type: BasicType,
    pub name: String,
    pub n_elements: usize,
    pub data: Vec<ProcparData>,
}

#[derive(Debug)]
enum ProcparData {
    String(String),
    Real(f64),
}

impl ProcparData {
    pub fn get_numeric(&self) -> Result<f64, AgilentFidError> {
        match &self {
            ProcparData::String(s) => {
                Ok(s.parse::<f64>().map_err(|_| AgilentFidError::ParseError)?)
            }
            ProcparData::Real(val) => Ok(*val),
        }
    }
    pub fn get_string(&self) -> String {
        match &self {
            ProcparData::String(s) => s.clone(),
            ProcparData::Real(val) => val.to_string(),
        }
    }
}

pub struct ProcPar {
    pub fields: IndexMap<String, ProcParField>,
}

impl ProcPar {
    pub fn open(procpar_file: impl AsRef<Path>) -> Result<Self, AgilentFidError> {

        let mut f = File::open(procpar_file).map_err(|_|AgilentFidError::ProcparNotFound)?;
        let mut s = String::new();
        f.read_to_string(&mut s).expect("failed to read procpar file");

        let field_re = Regex::new(r"^(?:(?:\w+|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+){10}(?:\w+|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$").expect("regex to be valid");
        let procpar_lines: Vec<&str> = s.lines().collect();
        let var_definition_index: Vec<usize> = procpar_lines
            .iter()
            .enumerate()
            .filter_map(|(idx, line)| {
                if field_re.is_match(line) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        let mut h = IndexMap::<String, ProcParField>::with_capacity(var_definition_index.len());
        var_definition_index.windows(2).for_each(|index| {
            let this_index = index[0];
            let next_index = index[1];
            let definition_elements: Vec<&str> =
                procpar_lines[this_index].split_whitespace().collect();
            let var_name = definition_elements[0].to_string();
            let var_sub_type =
                SubType::from_val(definition_elements[1].parse().expect("this to be parsable"));
            let var_basic_type =
                BasicType::from_val(definition_elements[2].parse().expect("this to be parsable"));
            let data_elements = procpar_lines[this_index + 1..next_index].join(" ");
            let data_elements: Vec<&str> = data_elements.split_whitespace().collect();
            // the first element is always an integer specifying the number of trailing elements
            let n_elements = data_elements[0]
                .parse::<usize>()
                .expect("this to be parsable");
            let field = match var_basic_type {
                BasicType::String | BasicType::Undefined => {
                    let var_data: Vec<ProcparData> = data_elements[1..n_elements + 1]
                        .iter()
                        .map(|mut s| {
                            let mut s = s.to_string();
                            s.retain(|c| !r#""'"#.contains(c));
                            ProcparData::String(s)
                        })
                        .collect();
                    ProcParField {
                        sub_type: var_sub_type,
                        basic_type: var_basic_type,
                        name: var_name.clone(),
                        n_elements,
                        data: var_data,
                    }
                }
                BasicType::Real => {
                    let var_data: Vec<ProcparData> = data_elements[1..n_elements + 1]
                        .iter()
                        .map(|mut s| {
                            let num = s.parse::<f64>().expect("this to be parsable");
                            ProcparData::Real(num)
                        })
                        .collect();
                    ProcParField {
                        sub_type: var_sub_type,
                        basic_type: var_basic_type,
                        name: var_name.clone(),
                        n_elements,
                        data: var_data,
                    }
                }
            };
            h.insert(var_name, field);
        });
        Ok(Self { fields: h })
    }

    fn to_hash(&self) -> IndexMap<String, String> {
        let mut h = IndexMap::<String, String>::new();
        for (key, val) in self.fields.iter() {
            let field_name = format!("z_Agilent_{}", key);
            if val.n_elements == 1 {
                h.insert(field_name, val.data[0].get_string());
            } else {
                let mut array_def = format!("{}:1,", val.n_elements);
                let elements: Vec<String> = val.data.iter().map(|p| p.get_string()).collect();
                let field_string = elements.join(" ");
                array_def.push_str(&field_string);
                h.insert(field_name, array_def);
            }
        }
        h
    }

    fn field_of_view_mm(&self) -> Result<[f32; 3], AgilentFidError> {
        let x = self
            .fields
            .get("lro")
            .ok_or(AgilentFidError::ProcparFieldError("lro".to_string()))?
            .data[0]
            .get_numeric()? as f32
            * 10.0;
        let y = self
            .fields
            .get("lpe")
            .ok_or(AgilentFidError::ProcparFieldError("lpe".to_string()))?
            .data[0]
            .get_numeric()? as f32
            * 10.0;
        let z = self
            .fields
            .get("lpe2")
            .ok_or(AgilentFidError::ProcparFieldError("lpe2".to_string()))?
            .data[0]
            .get_numeric()? as f32
            * 10.0;
        Ok([x, y, z])
    }

    fn matrix_size(&self) -> Result<[usize; 3], AgilentFidError> {
        let read = self.n_read()?;
        let v1 = self.n_phase1()?;
        let v2 = self.n_phase2();
        Ok([read, v1, v2])
    }

    fn n_views(&self) -> Result<usize, AgilentFidError> {
        let f = self.fields.get("nf").expect("field to exist");
        if let ProcparData::Real(n_views) = f.data[0] {
            Ok(n_views as usize)
        } else {
            Err(AgilentFidError::ProcparFieldError("nf".to_string()))
        }
    }

    fn cs_table_path(&self) -> Result<PathBuf, AgilentFidError> {
        let tab = self
            .fields
            .get("petableCS")
            .ok_or(AgilentFidError::CSTableNotFound)?;
        if let ProcparData::String(cs_table_path) = &tab.data[0] {
            Ok(PathBuf::from(cs_table_path))
        } else {
            Err(AgilentFidError::CSTableNotFound)?
        }
    }

    /// number of complex samples per trace
    fn n_read(&self) -> Result<usize, AgilentFidError> {
        let f = self.fields.get("np").expect("field to exist");
        if let ProcparData::Real(n_points) = f.data[0] {
            Ok(n_points as usize / 2)
        } else {
            Err(AgilentFidError::ProcparFieldError("np".to_string()))
        }
    }

    fn n_phase1(&self) -> Result<usize, AgilentFidError> {
        let f = self.fields.get("nv").expect("field to exist");
        if let ProcparData::Real(n_views) = f.data[0] {
            Ok(n_views as usize)
        } else {
            Err(AgilentFidError::ProcparFieldError("nv".to_string()))
        }
    }

    fn n_phase2(&self) -> usize {
        let f = self.fields.get("nv2").expect("field to exist");
        if let ProcparData::Real(n_views) = f.data[0] {
            n_views as usize
        } else {
            1
        }
    }
    
    fn convert_cs_table(&self, output_file: impl AsRef<Path>) -> Result<(), AgilentFidError> {
        let stream_tab = self.cs_stream()?;
        let mut s = stream_tab.iter().map(|val| val.to_string()).join("\n");
        s.push('\n');
        let mut f = File::create(output_file)?;
        f.write_all(s.as_bytes())?;
        Ok(())
    }

    fn cs_stream(&self) -> Result<Vec<i32>, AgilentFidError> {
        let tab = self.cs_table_path()?;
        if !tab.exists() {
            Err(AgilentFidError::CSTableNotFound)?
        }
        let mat_size = self.matrix_size()?;
        let dims = [mat_size[1] as i32, mat_size[2] as i32];

        let mut s = String::new();
        let mut f = File::open(&tab)?;
        f.read_to_string(&mut s)?;

        let mut stream_tab = Vec::<i32>::new();
        s.chars().enumerate().for_each(|(i, c)| {
            if c == '1' {
                let coord2 = (i as i32) % dims[0] - dims[0] / 2;
                let coord1 = (i as i32) / dims[1] - dims[1] / 2;
                stream_tab.push(coord2);
                stream_tab.push(coord1);
            }
        });
        Ok(stream_tab)
    }

    fn copy_cs_table(&self, output_file: &Path) -> Result<(), AgilentFidError> {
        let tab = self.cs_table_path()?;
        if !tab.exists() {
            Err(AgilentFidError::CSTableNotFound)?
        }
        std::fs::copy(&tab, output_file)
            .map_err(|_| AgilentFidError::UnableToCopyCSTable(tab, output_file.to_owned()))?;
        Ok(())
    }

    fn n_echoes(&self) -> Result<usize, AgilentFidError> {
        let f = self
            .fields
            .get("ne")
            .ok_or(AgilentFidError::ProcparFieldError("ne".to_string()))?;
        if let ProcparData::Real(n_echoes) = f.data[0] {
            Ok(n_echoes as usize)
        } else {
            Err(AgilentFidError::ProcparFieldError("ne".to_string()))
        }
    }
}