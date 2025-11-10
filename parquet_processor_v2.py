import boto3
import pandas as pd
import numpy as np
from typing import Iterator, List, Tuple
import os
from pathlib import Path
from tqdm import tqdm

# Your existing function and headers
HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    },
    "llama-3": {
        "magic": 20240801,
        "version": 7,
        "token_dtype": np.uint32,
    },
}

def write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    # construct the header
    header = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


class ParquetTokenProcessor:
    def __init__(self, bucket_name: str, s3_prefix: str = "", 
                 tokens_per_file: int = 10**8, model_desc: str = "gpt-2",
                 output_dir: str = "./output", temp_dir: str = "./temp"):
        """
        Initialize the processor.
        
        Args:
            bucket_name: S3 bucket containing parquet files
            s3_prefix: Prefix/folder path in S3 bucket
            tokens_per_file: Number of tokens per output file (default: 10^8)
            model_desc: Model descriptor for output format ("gpt-2" or "llama-3")
            output_dir: Directory to save output .bin files
            temp_dir: Directory for temporary parquet downloads
        """
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.tokens_per_file = tokens_per_file
        self.model_desc = model_desc
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
    def list_parquet_files(self) -> List[str]:
        """List all parquet files in the S3 bucket with the given prefix."""
        parquet_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.s3_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.parquet'):
                        parquet_files.append(obj['Key'])
        
        return sorted(parquet_files)
    
    def download_parquet_file(self, s3_key: str) -> str:
        """Download a parquet file from S3 to local temp directory."""
        local_filename = self.temp_dir / Path(s3_key).name
        
        self.s3_client.download_file(self.bucket_name, s3_key, str(local_filename))
        
        return str(local_filename)
    
    def read_tokens_from_parquet(self, parquet_file: str) -> Iterator[np.ndarray]:
        """
        Read tokens from a parquet file.
        
        Yields individual numpy arrays (documents) from the parquet file.
        Assumes each row contains a numpy array of tokens.
        """
        try:
            df = pd.read_parquet(parquet_file)
            
            # Assuming the tokens are in a column - adjust column name as needed
            # Common column names might be 'tokens', 'input_ids', 'data', etc.
            token_columns = [col for col in df.columns if 'token' in col.lower() or 'input' in col.lower() or 'data' in col.lower()]
            
            if not token_columns:
                # If no obvious token column, assume first column contains tokens
                token_column = df.columns[0]
            else:
                token_column = token_columns[0]
            
            for _, row in df.iterrows():
                tokens = row[token_column]
                
                # Convert to numpy array if it isn't already
                if not isinstance(tokens, np.ndarray):
                    tokens = np.array(tokens)
                
                yield tokens
                
        except Exception as e:
            raise
    
    def process_files(self, cleanup_temp: bool = True):
        """
        Main processing function that accumulates exactly 10^8 tokens before writing.
        
        Args:
            cleanup_temp: Whether to delete downloaded parquet files after processing
        """
        parquet_files = self.list_parquet_files()
        
        output_file_counter = 0
        total_processed_tokens = 0
        token_buffer = []
        current_buffer_size = 0
        
        # Create progress bar for parquet files
        pbar = tqdm(parquet_files, desc="Processing parquet files", unit="file")
        
        for s3_key in pbar:
            # Update progress bar description with current file
            pbar.set_postfix({
                'current': Path(s3_key).name[:30] + ('...' if len(Path(s3_key).name) > 30 else ''),
                'tokens': f"{total_processed_tokens:,}",
                'buffer': f"{current_buffer_size:,}",
                'files_out': output_file_counter
            })
            
            # Download parquet file
            local_parquet = self.download_parquet_file(s3_key)
            
            try:
                # Process tokens from this file
                for token_array in self.read_tokens_from_parquet(local_parquet):
                    # Convert to list for easier manipulation
                    tokens = token_array.tolist() if isinstance(token_array, np.ndarray) else list(token_array)
                    
                    # Add tokens to buffer
                    token_buffer.extend(tokens)
                    current_buffer_size += len(tokens)
                    
                    # Check if we have enough tokens to write a file
                    while current_buffer_size >= self.tokens_per_file:
                        # Extract exactly tokens_per_file tokens
                        tokens_to_write = token_buffer[:self.tokens_per_file]
                        token_buffer = token_buffer[self.tokens_per_file:]
                        current_buffer_size -= self.tokens_per_file
                        
                        # Write the file with appropriate naming
                        if output_file_counter == 0:
                            output_filename = self.output_dir / "fineweb_val_000000.bin"
                        else:
                            output_filename = self.output_dir / f"fineweb_train_{output_file_counter - 1:06d}.bin"
                        
                        write_datafile(str(output_filename), tokens_to_write, self.model_desc)
                        
                        total_processed_tokens += len(tokens_to_write)
                        output_file_counter += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'current': Path(s3_key).name[:30] + ('...' if len(Path(s3_key).name) > 30 else ''),
                            'tokens': f"{total_processed_tokens:,}",
                            'buffer': f"{current_buffer_size:,}",
                            'files_out': output_file_counter
                        })
            
            finally:
                # Clean up downloaded parquet file
                if cleanup_temp:
                    os.remove(local_parquet)
        
        pbar.close()
        
        # Write any remaining tokens to a final file
        if token_buffer:
            if output_file_counter == 0:
                output_filename = self.output_dir / "fineweb_val_000000.bin"
            else:
                output_filename = self.output_dir / f"fineweb_train_{output_file_counter - 1:06d}.bin"
            
            write_datafile(str(output_filename), token_buffer, self.model_desc)
            total_processed_tokens += len(token_buffer)


# Example usage
def main():
    # Configure your S3 details
    BUCKET_NAME = "your-bucket-name"
    S3_PREFIX = "path/to/parquet/files/"  # Optional prefix/folder
    MODEL_DESC = "gpt-2"  # or "llama-3"
    OUTPUT_DIR = "./token_files"
    TEMP_DIR = "./temp_parquet"
    
    # Create processor
    processor = ParquetTokenProcessor(
        bucket_name=BUCKET_NAME,
        s3_prefix=S3_PREFIX,
        tokens_per_file=10**8,
        model_desc=MODEL_DESC,
        output_dir=OUTPUT_DIR,
        temp_dir=TEMP_DIR
    )
    
    # Process all files
    processor.process_files(cleanup_temp=True)


if __name__ == "__main__":
    main()