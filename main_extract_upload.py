import argparse
import os
import re
import subprocess
import sys
from minio import Minio
from minio.error import S3Error

# --- Prewritten Configuration & Credentials ---
ENDPOINT = "68.168.30.34:9000"
ACCESS_KEY = "ProtocolPinAdmin"
SECRET_KEY = "ProtocolPin@Storage2026!"


# ----------------------------------------------

def parse_mapping_file(file_path):
    """Parses the mapping text file to extract folder, pdf name, and full remote s3 path."""
    tasks = []
    current_folder = None
    
    folder_regex = re.compile(r"^Folder:\s+(.+)/?$")
    file_regex = re.compile(r"^\s+└──\s+(.+\.pdf)\s+\(([^)]+)\)")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            folder_match = folder_regex.match(line)
            if folder_match:
                current_folder = folder_match.group(1).strip("/")
                continue
                
            file_match = file_regex.match(line)
            if file_match and current_folder:
                pdf_filename = file_match.group(1).strip()
                remote_s3_path = file_match.group(2).strip()
                tasks.append({
                    "folder": current_folder,
                    "pdf_filename": pdf_filename,
                    "remote_s3_path": remote_s3_path
                })
                
    return tasks

def log_error(error_log_path, message):
    """Appends error messages to an error log file."""
    with open(error_log_path, "a", encoding="utf-8") as log_f:
        log_f.write(message + "\n")

def main():
    parser = argparse.ArgumentParser(description="End-to-end MinIO PDF to Markdown batch processor with custom CLI arguments.")
    parser.add_argument("bucket", help="Name of the MinIO bucket (e.g., stmicroelectronics)")
    parser.add_argument("main_folder", help="Main folder/category inside bucket (e.g., rm)")
    parser.add_argument("object_name", help="Specific object/subfolder prefix if filtering further (or use '.' if mapping file handles full paths)")
    parser.add_argument("mapping_file", help="Path to the mapping text file (e.g., st_rm.txt)")
    parser.add_argument("--local-workdir", default="./work_dir", help="Local temporary working directory")
    parser.add_argument("--error-log", default="error_log.txt", help="Path to error log file")

    args = parser.parse_args()

    # Initialize MinIO Client
    client = Minio(
        ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
    )

    if not os.path.exists(args.mapping_file):
        print(f"Error: Mapping file '{args.mapping_file}' not found.", file=sys.stderr)
        sys.exit(1)

    tasks = parse_mapping_file(args.mapping_file)
    print(f"Found {len(tasks)} tasks to process from mapping file.")

    os.makedirs(args.local_workdir, exist_ok=True)

    success_count = 0
    fail_count = 0

    for idx, task in enumerate(tasks, start=1):
        pdf_filename = task["pdf_filename"]
        raw_s3_path = task["remote_s3_path"]
        
        # Strip leading bucket name if included in the logged path string
        if raw_s3_path.startswith(args.bucket + "/"):
            object_s3_path = raw_s3_path[len(args.bucket) + 1:]
        else:
            object_s3_path = raw_s3_path

        # Optional alignment check with CLI provided main_folder / object_name if needed
        # The exact path inside parentheses of mapping file takes precedence for precision.
        remote_dir = os.path.dirname(object_s3_path)
        base_name_no_ext = os.path.splitext(pdf_filename)[0]

        print(f"\n[{idx}/{len(tasks)}] Processing: {pdf_filename}")

        # Local working file paths
        local_pdf_path = os.path.join(args.local_workdir, pdf_filename)
        local_md_filename = f"{base_name_no_ext}.md"
        local_md_path = os.path.join(args.local_workdir, local_md_filename)
        local_img_dir = os.path.join(args.local_workdir, "extracted_img")

        try:
            # Step 1: Download PDF from MinIO
            print(f"  -> Downloading from MinIO path: {object_s3_path}")
            client.fget_object(args.bucket, object_s3_path, local_pdf_path)

            # Step 2: Run external conversion script (no timeout)
            print(f"  -> Converting PDF to Markdown...")
            cmd = [
                "python3", "main.py",
                local_pdf_path,
                "-o", local_md_path,
                "--img-dir", local_img_dir,
                "--dpi", "150"
            ]
            subprocess.run(cmd, check=True)

            # Step 3: Upload generated Markdown file back to MinIO
            remote_md_path = f"{remote_dir}/{local_md_filename}" if remote_dir else local_md_filename
            print(f"  -> Uploading Markdown to MinIO: {remote_md_path}")
            client.fput_object(args.bucket, remote_md_path, local_md_path)

            # Step 4: Upload extracted images directory contents back to MinIO
            if os.path.exists(local_img_dir) and os.path.isdir(local_img_dir):
                print(f"  -> Uploading extracted images...")
                for root, _, files in os.walk(local_img_dir):
                    for img_file in files:
                        local_img_file_path = os.path.join(root, img_file)
                        rel_path = os.path.relpath(local_img_file_path, args.local_workdir)
                        remote_img_path = f"{remote_dir}/{rel_path.replace(os.sep, '/')}" if remote_dir else rel_path.replace(os.sep, '/')
                        
                        client.fput_object(args.bucket, remote_img_path, local_img_file_path)

            # Step 5: Cleanup local temporary files for this job
            print(f"  -> Cleaning up local temporary files...")
            if os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)
            if os.path.exists(local_md_path):
                os.remove(local_md_path)
            if os.path.exists(local_img_dir):
                for root, dirs, files in os.walk(local_img_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(local_img_dir)

            success_count += 1
            print(f"  -> Successfully completed: {pdf_filename}")

        except Exception as e:
            fail_count += 1
            error_msg = f"[ERROR] Task {idx} ({pdf_filename}): {str(e)}"
            print(f"  {error_msg}")
            log_error(args.error_log, error_msg)

            # Attempt basic cleanup on failure
            try:
                if os.path.exists(local_pdf_path):
                    os.remove(local_pdf_path)
                if os.path.exists(local_md_path):
                    os.remove(local_md_path)
                if os.path.exists(local_img_dir):
                    for root, dirs, files in os.walk(local_img_dir, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(local_img_dir)
            except:
                pass

    print(f"\nBatch processing finished.")
    print(f"Successful: {success_count} | Failed: {fail_count}")
    if fail_count > 0:
        print(f"Check '{args.error_log}' for failure details.")

if __name__ == "__main__":
    main()