import sys
import os

EMPTY_THRESHOLD = 2048 

def check_missing(input_file, output_file):
    missing_entries = []
    
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            
            base_path = parts[3]
            if base_path.endswith('.root'):
                base_path = base_path[:-5]

            expected_file = base_path + "_0.parquet" 

            is_valid = False
            if os.path.exists(expected_file):
                if os.path.getsize(expected_file) > EMPTY_THRESHOLD:
                    is_valid = True
            
            if not is_valid:
                missing_entries.append(line)

    with open(output_file, 'w') as f:
        f.writelines(missing_entries)
    
    print(f"Done. Found {len(missing_entries)} files that are missing or empty. Written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extractMissingOrEmpty.py <input_args.txt> <output_args.txt>")
    else:
        check_missing(sys.argv[1], sys.argv[2])
