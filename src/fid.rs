use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use byteorder::{BigEndian, ByteOrder, ReadBytesExt};
use num_complex::Complex32;
use serde::{Deserialize, Serialize};
use crate::{AgilentFidError, FILE_HEADER_SIZE_BYTES};

#[derive(Debug, Serialize, Deserialize)]
pub struct FileHeader {
    pub n_blocks: usize,
    pub n_traces: usize,
    pub n_points: usize,
    pub bytes_per_element: usize,
    pub bytes_per_trace: usize,
    pub bytes_per_block: usize,
    pub version_id: usize,
    pub acq_status: AcqStatus,
    pub n_block_headers_per_block: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AcqStatus {
    pub has_data: bool,
    pub is_spectrum: bool,
    pub is_int32: bool,
    pub is_float32: bool,
    pub s_complex: bool,
    pub s_hypercomplex: bool,
    pub unused6: bool,
    pub s_acqpar: bool,
    pub s_secnd: bool,
    pub s_transf: bool,
    pub s_np: bool,
    pub unused10: bool,
    pub s_nf: bool,
    pub s_ni: bool,
    pub s_ni2: bool,
    pub unused15: bool,
}

#[derive(Debug)]
pub struct BlockStatus {
    pub has_data: bool,
    pub is_spectrum: bool,
    pub is_int32: bool,
    pub is_float32: bool,
    pub s_complex: bool,
    pub s_hypercomplex: bool,
    pub unused6: bool,
    pub more_blocks: bool,
    pub np_cmplx: bool,
    pub nf_cmplx: bool,
    pub ni_cmplx: bool,
    pub ni2_cmplx: bool,
    pub unused12: bool,
    pub unused13: bool,
    pub unused14: bool,
    pub unused15: bool,
}

#[derive(Debug)]
struct BlockHeader {
    pub scale: i16,
    pub status: BlockStatus,
    pub index: i16,
    pub mode: i16,
    pub ct_count: i32,
    pub lpval: f32,
    pub rpval: f32,
    pub lvl: f32,
    pub tlt: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgilentFidHeader {
    pub path: PathBuf,
    pub file_header: FileHeader,
    pub expected_file_size: usize,
}

pub struct AgilentSubFid {
    bytes: Vec<u8>,
}

impl AgilentFidHeader {

    pub fn open(fid_file: impl AsRef<Path>) -> Result<Self, AgilentFidError> {

        let mut f = File::open(&fid_file).map_err(|_| AgilentFidError::FidFileNotFound)?;

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
            path: fid_file.as_ref().to_path_buf(),
            file_header,
            expected_file_size: FILE_HEADER_SIZE_BYTES + n_blocks * bytes_per_block,
        })
    }

    pub fn extract_data(&self,index_values: &[usize]) -> Result<Vec<Complex32>, AgilentFidError > {
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

    pub fn read_sub_fid(fid_file: impl AsRef<Path>) -> Result<AgilentSubFid, AgilentFidError> {
        let mut f = File::open(fid_file)?;
        let mut bytes = vec![];
        f.read_to_end(&mut bytes).expect("to read file");
        Ok(AgilentSubFid { bytes })
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

    // fn write_sub_fid(
    //     &self,
    //     filename: &Path,
    //     vol_index: usize,
    //     n_totoal_vols: usize,
    //     multi_echo_mode: bool,
    // ) -> Result<(), AgilentFidError> {
    //     if multi_echo_mode {
    //         let block_index = 0;
    //         let (bytes, header) = self.read_block(block_index)?;
    //         if !header.status.is_float32 {
    //             Err(AgilentFidError::Int32NotSupported)?
    //         }
    //
    //         let view_size = self.file_header.bytes_per_trace;
    //         let views: Vec<&[u8]> = bytes.chunks_exact(view_size).collect();
    //         let n_views_per_object = views.len() / n_totoal_vols;
    //
    //         let mut f = File::create(filename).map_err(|_| {
    //             AgilentFidError::UnableToCreateFile(filename.to_string_lossy().to_string())
    //         })?;
    //         for i in 0..n_views_per_object {
    //             let view_idx = n_totoal_vols * i + vol_index;
    //             let byte_slice = views
    //                 .get(view_idx)
    //                 .ok_or(AgilentFidError::CannotFillBlockBuffer)?;
    //             f.write_all(*byte_slice).map_err(|_| {
    //                 AgilentFidError::UnableToCreateFile(filename.to_string_lossy().to_string())
    //             })?;
    //         }
    //     } else {
    //         let block_index = vol_index;
    //         let (bytes, header) = self.read_block(block_index)?;
    //         if header.status.is_float32 {
    //             let mut f = File::create(filename).map_err(|_| {
    //                 AgilentFidError::UnableToCreateFile(filename.to_string_lossy().to_string())
    //             })?;
    //             f.write_all(&bytes).map_err(|_| {
    //                 AgilentFidError::UnableToCreateFile(filename.to_string_lossy().to_string())
    //             })?;
    //         } else {
    //             Err(AgilentFidError::Int32NotSupported)?
    //         }
    //     }
    //
    //     Ok(())
    // }

    pub fn read_block(&self, block_idx: usize) -> Result<(Vec<u8>, BlockHeader), AgilentFidError> {
        if block_idx >= self.file_header.n_blocks {
            Err(AgilentFidError::BlockIndexOutOfRange)?
        }

        let mut f = File::open(&self.path).expect("file to exist");

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
            Err(AgilentFidError::DataBlockNotReady(self.path.clone()))?
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
