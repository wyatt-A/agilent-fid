use std::io;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

pub mod procpar;
pub mod fid;

const FILE_HEADER_SIZE_BYTES: usize = 32;

#[derive(Debug, Serialize, Deserialize)]
pub enum AgilentFidError {
    BlockIndexOutOfRange,
    DataBlockNotReady(PathBuf),
    CannotFillBlockBuffer,
    CSTableNotFound,
    UnableToCreateFile(String),
    UnableToCreateDir(String),
    Int32NotSupported,
    ProcparNotFound,
    FidFileNotFound,
    ProcparFieldError(String),
    ProcparFieldIndexError(String),
    ParseError,
    UnableToCleanup(String),
    UnableToCopyCSTable(PathBuf, PathBuf),
    UnableToConvertCSMask(PathBuf),
    FailedToDetermineMatrixSizeFromCSMaskName(PathBuf),
    IO(String),
}

impl From<io::Error> for AgilentFidError {
    fn from(value: io::Error) -> Self {
        AgilentFidError::IO(value.to_string())
    }
}