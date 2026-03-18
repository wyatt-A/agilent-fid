use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cs_table::{cs_table::CSTable, ViewTable};
use headfile::headfile::{AcqHeadfileParams, DWHeadfileParams, Headfile};
use itertools::Itertools;
use ndarray::{Array1, Array3, ArrayD, ShapeBuilder};
use num_complex::{Complex, Complex32};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{create_dir_all, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crate::{
    cfl::Cfl,
    dimensions::{DimensionLabel, DimensionOrder},
    mr_raw::ToMrRaw,
    phase_encoding::PhaseEncodingMethod,
    raw_to_kspace::{RawToKspaceParams, RAW_TO_KSPACE_FILENAME},
};



#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Success(String),
    Error(AgilentFidError),
}



impl ToMrRaw for AgilentSubFid {
    fn complex_stream(&self) -> Vec<Complex<f32>> {
        let n_elems = self.bytes.len() / 4;
        let mut out = Vec::<Complex<f32>>::with_capacity(n_elems / 2);
        let mut floats = vec![0.0; n_elems];
        BigEndian::read_f32_into(&self.bytes, &mut floats);
        for i in 0..n_elems / 2 {
            let val = Complex::new(floats[2 * i], floats[2 * i + 1]);
            out.push(val);
        }
        out
    }
}

impl AgilentFid {

    pub fn extract_data(&self,index_values: &[usize]) -> Result<Vec<Complex<f32>>,AgilentFidError> {
        let mut block_table = HashMap::<usize,Vec<Complex32>>::new();
        let mut data = vec![];
        for &idx in index_values {
            let block_idx = self.elem_idx_to_block_idx(idx);
            if !block_table.contains_key(&block_idx) {
                let (block_data,header) = self.read_block(block_idx)?;
                if !header.status.is_float32 {
                    Err(AgilentFidError::Int32NotSupported)?
                }
                let block_values:Vec<_> = block_data.chunks_exact(4*2).map(|byte_chunk|{
                    Complex32::new(
                        BigEndian::read_f32(&byte_chunk[0..4]),
                        BigEndian::read_f32(&byte_chunk[4..])
                    )
                }).collect();
                block_table.insert(block_idx, block_values);
            }
            let block_values = block_table.get(&block_idx).unwrap().as_slice();
            let lidx = self.elem_idx_to_local_idx(idx);
            data.push(
                block_values[lidx]
            )
        }
        Ok(data)
    }

    pub fn meta_data(&self,vol_index: usize) -> Result<HashMap<String,String>,AgilentFidError> {
        let proc = ProcPar::open(&self.procpar_file);
        proc.meta_data(vol_index)
    }

    pub fn view_table(&self) -> Result<ViewTable,AgilentFidError> {
        let proc = ProcPar::open(&self.procpar_file);
        proc.view_table()
    }

    #[inline(always)]
    // get the block-local index
    fn elem_idx_to_local_idx(&self,idx:usize) -> usize {
        idx % self.complex_points_per_block()
    }

    #[inline(always)]
    fn complex_points_per_block(&self) -> usize {
        let data_bytes_per_block = self.file_header.bytes_per_trace * self.file_header.n_traces;
        data_bytes_per_block / (self.file_header.bytes_per_element * 2)
    }

    #[inline(always)]
    // this index is a single complex value
    fn elem_idx_to_block_idx(&self,idx:usize) -> usize {
        let bytes_per_idx = self.file_header.bytes_per_element * 2; // complex-valued
        idx * bytes_per_idx / self.file_header.bytes_per_block
    }

    pub fn read_sub_fid(fid_file: &Path) -> Result<AgilentSubFid, AgilentFidError> {
        let mut f = File::open(fid_file).map_err(|_| AgilentFidError::FidFileNotFound)?;
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).expect("to read file");
        Ok(AgilentSubFid { bytes })
    }

    pub fn open(fid_file: &Path, procpar_file: PathBuf) -> Result<Self, AgilentFidError> {
        if !procpar_file.exists() {
            Err(AgilentFidError::ProcparNotFound)?
        }

        let mut f = File::open(fid_file).map_err(|_| AgilentFidError::FidFileNotFound)?;

        let n_blocks = f.read_i32::<BigEndian>().unwrap() as usize;
        let n_traces = f.read_i32::<BigEndian>().unwrap() as usize;
        let n_points = f.read_i32::<BigEndian>().unwrap() as usize;
        let bytes_per_element = f.read_i32::<BigEndian>().unwrap() as usize;
        let bytes_per_trace = f.read_i32::<BigEndian>().unwrap() as usize;
        let bytes_per_block = f.read_u32::<BigEndian>().unwrap() as usize;
        let version_id = f.read_i16::<BigEndian>().unwrap() as usize;
        let acq_status = f.read_i16::<BigEndian>().unwrap();
        let n_block_headers_per_block = f.read_i32::<BigEndian>().unwrap() as usize;

        if n_block_headers_per_block != 1 {
            panic!("multiple block headers per block is not supported");
        }

        let acq_status = AcqStatus {
            has_data: (acq_status & 1) == 1,
            is_spectrum: (acq_status >> 1 & 1) == 1,
            is_int32: (acq_status >> 2 & 1) == 1,
            is_float32: (acq_status >> 3 & 1) == 1,
            s_complex: (acq_status >> 4 & 1) == 1,
            s_hypercomplex: (acq_status >> 5 & 1) == 1,
            unused6: (acq_status >> 6 & 1) == 1,
            s_acqpar: (acq_status >> 7 & 1) == 1,
            s_secnd: (acq_status >> 8 & 1) == 1,
            s_transf: (acq_status >> 9 & 1) == 1,
            s_np: (acq_status >> 10 & 1) == 1,
            unused10: (acq_status >> 11 & 1) == 1,
            s_nf: (acq_status >> 12 & 1) == 1,
            s_ni: (acq_status >> 13 & 1) == 1,
            s_ni2: (acq_status >> 14 & 1) == 1,
            unused15: (acq_status >> 15 & 1) == 1,
        };

        let file_header = FileHeader {
            n_blocks,
            n_traces,
            n_points,
            bytes_per_element,
            bytes_per_trace,
            bytes_per_block,
            version_id,
            acq_status,
            n_block_headers_per_block,
        };

        Ok(Self {
            file_header,
            expected_file_size: FILE_HEADER_SIZE_BYTES + n_blocks * bytes_per_block,
            file_path: fid_file.to_owned(),
            procpar_file,
        })
    }

    pub fn fid_prep(
        &self,
        out_dir: &Path,
        vol_index: usize,
        total_vols: usize,
        multi_echo_mode: bool,
        dont_convert_cs_table: bool,
        view_acceleration: usize,
        dummy_excitations: usize,
    ) -> Result<PathBuf, AgilentFidError> {
        let m_number = civm_rust_utils::m_number(vol_index, total_vols);
        let out_dir = out_dir.join(&m_number);
        if !out_dir.exists() {
            create_dir_all(&out_dir).map_err(|_| {
                AgilentFidError::UnableToCreateDir(out_dir.to_string_lossy().to_string())
            })?;
        }
        std::fs::copy(&self.procpar_file, &out_dir.join("procpar"))
            .map_err(|_| AgilentFidError::ProcparNotFound)?;
        let out_fid = out_dir.join(&m_number).with_extension("fid");
        let out_cs_table = out_dir.join("cs_table");
        let out_meta = out_dir.join("meta.txt");
        let rtk = out_dir.join(RAW_TO_KSPACE_FILENAME).with_extension("json");
        let ac_file = out_dir.join(&m_number);
        let p = ProcPar::open(&self.procpar_file);

        // cs table is either a binary mask, or a stream table. We want a stream table
        if dont_convert_cs_table {
            p.copy_cs_table(&out_cs_table)?;
        } else {
            p.convert_cs_table(&out_cs_table)?;
        }

        let cs_tab = CSTable::open(&out_cs_table);
        let view_coords = cs_tab.coordinate_pairs_i16();

        self.write_raw_to_kspace_params(
            &rtk,
            total_vols,
            multi_echo_mode,
            view_acceleration,
            dummy_excitations,
            view_coords,
        )?;
        p.write_meta_data(&out_meta, vol_index)?;
        self.write_sub_fid(&out_fid, vol_index, total_vols, multi_echo_mode)?;
        civm_rust_utils::write_to_file(
            &ac_file,
            Some("ac"),
            &format!("creation_date={}", civm_rust_utils::time_stamp()),
        );
        Ok(out_dir)
    }

    fn write_raw_to_kspace_params(
        &self,
        output_file: &Path,
        total_vols: usize,
        multi_echo_mode: bool,
        view_acceleration: usize,
        dummy_excitations: usize,
        view_coords: Vec<[i16; 2]>,
    ) -> Result<(), AgilentFidError> {
        let p = ProcPar::open(&self.procpar_file);
        let n_read = self.file_header.n_points / 2;

        let encoding_method =
            PhaseEncodingMethod::new(dummy_excitations, view_acceleration, &[0, 0], view_coords);

        let rtk = if multi_echo_mode {
            let n_views = self.file_header.n_traces / total_vols;
            RawToKspaceParams {
                dimension_order: DimensionOrder::from_vec(&[
                    DimensionLabel::Samples(n_read),
                    DimensionLabel::Views(n_views),
                ]),
                encoding_method,
                n_objects: 1,
                image_size: p.matrix_size()?,
            }
        } else {
            let n_views = self.file_header.n_traces;
            RawToKspaceParams {
                dimension_order: DimensionOrder::from_vec(&[
                    DimensionLabel::Samples(n_read),
                    DimensionLabel::Views(n_views),
                ]),
                encoding_method,
                n_objects: 1,
                image_size: p.matrix_size()?,
            }
        };

        rtk.to_file(output_file);
        Ok(())
    }

    pub fn dump_to_cfl(
        &self,
        vol_index: usize,
        n_total_vols: usize,
        multi_echo_mode: bool,
    ) -> Result<Cfl, AgilentFidError> {
        let sf = self.extract_sub_fid(vol_index, n_total_vols, multi_echo_mode)?;
        let stream = sf.complex_stream();
        let array = Array1::from_shape_vec((stream.len()).f(), stream).unwrap();
        Ok(Cfl::from_array(array.into_dyn()))
    }

    fn extract_sub_fid(
        &self,
        vol_index: usize,
        n_total_vols: usize,
        multi_echo_mode: bool,
    ) -> Result<AgilentSubFid, AgilentFidError> {
        let bytes = if multi_echo_mode {
            let block_index = 0;
            let (bytes, header) = self.read_block(block_index)?;
            if !header.status.is_float32 {
                Err(AgilentFidError::Int32NotSupported)?
            }

            let view_size = self.file_header.bytes_per_trace;
            let views: Vec<&[u8]> = bytes.chunks_exact(view_size).collect();
            let n_views_per_object = views.len() / n_total_vols;

            let mut byte_buff: Vec<u8> = vec![];
            for i in 0..n_views_per_object {
                let view_idx = n_total_vols * i + vol_index;
                let byte_slice = views
                    .get(view_idx)
                    .ok_or(AgilentFidError::CannotFillBlockBuffer)?;
                byte_slice
                    .into_iter()
                    .for_each(|byte| byte_buff.push(*byte));
            }
            byte_buff
        } else {
            let block_index = vol_index;
            let (bytes, header) = self.read_block(block_index)?;
            if header.status.is_float32 {
                bytes
            } else {
                Err(AgilentFidError::Int32NotSupported)?
            }
        };
        Ok(AgilentSubFid { bytes })
    }

    fn write_sub_fid(
        &self,
        filename: &Path,
        vol_index: usize,
        n_totoal_vols: usize,
        multi_echo_mode: bool,
    ) -> Result<(), AgilentFidError> {
        if multi_echo_mode {
            let block_index = 0;
            let (bytes, header) = self.read_block(block_index)?;
            if !header.status.is_float32 {
                Err(AgilentFidError::Int32NotSupported)?
            }

            let view_size = self.file_header.bytes_per_trace;
            let views: Vec<&[u8]> = bytes.chunks_exact(view_size).collect();
            let n_views_per_object = views.len() / n_totoal_vols;

            let mut f = File::create(filename).map_err(|_| {
                AgilentFidError::UnableToCreateFile(filename.to_string_lossy().to_string())
            })?;
            for i in 0..n_views_per_object {
                let view_idx = n_totoal_vols * i + vol_index;
                let byte_slice = views
                    .get(view_idx)
                    .ok_or(AgilentFidError::CannotFillBlockBuffer)?;
                f.write_all(*byte_slice).map_err(|_| {
                    AgilentFidError::UnableToCreateFile(filename.to_string_lossy().to_string())
                })?;
            }
        } else {
            let block_index = vol_index;
            let (bytes, header) = self.read_block(block_index)?;
            if header.status.is_float32 {
                let mut f = File::create(filename).map_err(|_| {
                    AgilentFidError::UnableToCreateFile(filename.to_string_lossy().to_string())
                })?;
                f.write_all(&bytes).map_err(|_| {
                    AgilentFidError::UnableToCreateFile(filename.to_string_lossy().to_string())
                })?;
            } else {
                Err(AgilentFidError::Int32NotSupported)?
            }
        }

        Ok(())
    }

    fn read_block(&self, block_idx: usize) -> Result<(Vec<u8>, BlockHeader), AgilentFidError> {
        if block_idx >= self.file_header.n_blocks {
            Err(AgilentFidError::BlockIndexOutOfRange)?
        }

        let mut f = File::open(&self.file_path).expect("file to exist");

        // this includes the block header
        let total_bytes_per_block = self.file_header.bytes_per_block;
        let data_bytes_per_block = self.file_header.bytes_per_trace * self.file_header.n_traces;
        let block_header_size = total_bytes_per_block - data_bytes_per_block;

        f.seek(SeekFrom::Start(
            (total_bytes_per_block * block_idx + FILE_HEADER_SIZE_BYTES) as u64,
        ))
            .expect("seek failed");

        let scale = f.read_i16::<BigEndian>().unwrap();
        let status = f.read_i16::<BigEndian>().unwrap();
        let index = f.read_i16::<BigEndian>().unwrap();
        let mode = f.read_i16::<BigEndian>().unwrap();
        let ct_count = f.read_i32::<BigEndian>().unwrap();
        let lpval = f.read_f32::<BigEndian>().unwrap();
        let rpval = f.read_f32::<BigEndian>().unwrap();
        let lvl = f.read_f32::<BigEndian>().unwrap();
        let tlt = f.read_f32::<BigEndian>().unwrap();

        let block_status = BlockStatus {
            has_data: (status & 1) == 1,
            is_spectrum: (status >> 1 & 1) == 1,
            is_int32: (status >> 2 & 1) == 1,
            is_float32: (status >> 3 & 1) == 1,
            s_complex: (status >> 4 & 1) == 1,
            s_hypercomplex: (status >> 5 & 1) == 1,
            unused6: (status >> 6 & 1) == 1,
            more_blocks: (status >> 7 & 1) == 1,
            np_cmplx: (status >> 8 & 1) == 1,
            nf_cmplx: (status >> 9 & 1) == 1,
            ni_cmplx: (status >> 10 & 1) == 1,
            ni2_cmplx: (status >> 11 & 1) == 1,
            unused12: (status >> 12 & 1) == 1,
            unused13: (status >> 13 & 1) == 1,
            unused14: (status >> 14 & 1) == 1,
            unused15: (status >> 15 & 1) == 1,
        };

        let block_header = BlockHeader {
            scale,
            status: block_status,
            index,
            mode,
            ct_count,
            lpval,
            rpval,
            lvl,
            tlt,
        };

        if !block_header.status.has_data {
            Err(AgilentFidError::DataBlockNotReady(self.file_path.clone()))?
        }

        let mut block_buff: Vec<u8> = vec![0; data_bytes_per_block];
        f.read_exact(&mut block_buff)
            .map_err(|_| AgilentFidError::CannotFillBlockBuffer)?;

        Ok((block_buff, block_header))
    }

    /// return fid info as json string
    pub fn info(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }
}



mod tests {
    use super::*;
    #[test]
    fn bvalue_extraction() {
        let p = ProcPar::open(Path::new("/home/wa41/recon_test_data/ser27.fid/procpar"));
        for vol_index in 0..7 {
            let path = format!("/home/wa41/recon_test_data/ser27.fid/meta{}.txt", vol_index);
            p.write_meta_data(Path::new(&path), vol_index)
                .expect("encountered an error");
        }
    }

    //cargo test --package mr_data --lib -- agilent_fid::tests::data_extraction --exact --nocapture
    #[test]
    fn data_extraction() {

        let fid_file = Path::new("/Users/Wyatt/scratch/ser11.fid/fid");
        let procpar = Path::new("/Users/Wyatt/scratch/ser11.fid/procpar");

        let af = AgilentFid::open(fid_file, procpar.to_path_buf()).unwrap();

        let meta = af.meta_data(0).unwrap();

        let view_tab = af.view_table().unwrap();

        println!("meta: {:?}",meta.get("fovy"));


        println!("complex points per block: {}",af.complex_points_per_block());

        println!("elem idx: {}",af.elem_idx_to_local_idx(2800));

        println!("{}",af.elem_idx_to_block_idx(2880))

    }


}

pub fn cs_mask_to_view_table(cs_mask_file:impl AsRef<Path>) -> Result<ViewTable, AgilentFidError> {
    if !cs_mask_file.as_ref().exists() {
        Err(AgilentFidError::CSTableNotFound)?
    }

    let re = Regex::new(r"CS(\d+)_").unwrap();
    let mat_size = if let Some(caps) = re.captures(&cs_mask_file.as_ref().file_name().unwrap().to_string_lossy()) {
        if let Some(matched) = caps.get(1) {
            matched.as_str().parse::<i32>().map_err(|_|AgilentFidError::FailedToDetermineMatrixSizeFromCSMaskName(cs_mask_file.as_ref().to_path_buf()))?
        }else {
            Err(AgilentFidError::FailedToDetermineMatrixSizeFromCSMaskName(cs_mask_file.as_ref().to_path_buf()))?
        }
    }else {
        Err(AgilentFidError::FailedToDetermineMatrixSizeFromCSMaskName(cs_mask_file.as_ref().to_path_buf()))?
    };

    let s = civm_rust_utils::read_to_string(cs_mask_file.as_ref(), None);
    let mut stream_tab = Vec::<i32>::new();
    s.chars().enumerate().for_each(|(i, c)| {
        if c == '1' {
            let coord2 = (i as i32) % mat_size - mat_size / 2;
            let coord1 = (i as i32) / mat_size - mat_size / 2;
            stream_tab.push(coord2);
            stream_tab.push(coord1);
        }
    });

    ViewTable::from_stream(&stream_tab)
        .map_err(|_|AgilentFidError::UnableToConvertCSMask(cs_mask_file.as_ref().to_path_buf()))
}