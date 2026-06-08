import ROOT
import os
import sys
import csv

# Suppress ROOT dictionary warnings to keep standard output clean
ROOT.gErrorIgnoreLevel = ROOT.kError

def process_file(filepath, csv_writer):
    """Processes a single ROOT file and writes results to the CSV writer."""
    f = ROOT.TFile.Open(filepath, "READ")
    if not f or f.IsZombie():
        # Print errors to stderr so they don't break your workflow
        print(f"Error: Cannot open file {filepath}", file=sys.stderr)
        if f: f.Close()
        return

    tree = f.Get("rawConv") or f.Get("cbmsim")
    if not tree:
        print(f"Error: No rawConv or cbmsim tree found in {filepath}", file=sys.stderr)
        f.Close()
        return

    # Disable heavy branches, enable only the header
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus("EventHeader*", 1)

    n_entries = tree.GetEntries()
    if n_entries == 0:
        f.Close()
        return

    # Load the first entry to get the run ID
    tree.GetEntry(0)
    run = int(tree.GetLeaf("EventHeader.fRunId").GetValue())
    
    # Cache the leaf pointer before the loop for maximum speed
    leaf_ev = tree.GetLeaf("EventHeader.fEventNumber")

    # Fast iteration over all events in the file
    for i_entry in range(n_entries):
        tree.GetEntry(i_entry)
        event_number = int(leaf_ev.GetValue())
        
        # Write directly to the CSV file
        csv_writer.writerow([run, event_number])

    # Explicitly close the file to free memory and prevent EOS connection limits
    f.Close()

def main():
    base_dir = "/eos/experiment/sndlhc/users/odurhan/multi_muon_search"
    output_filename = "tridents.csv"
    
    if not os.path.exists(base_dir):
        print(f"Fatal Error: Base directory {base_dir} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    # Open the CSV file once at the beginning
    counter = 0
    with open(output_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header row
        csv_writer.writerow(["run", "event_number"])
        
        # Traverse the Year directories (e.g., 2022, 2023)
        for year in os.listdir(base_dir):
            year_path = os.path.join(base_dir, year)
            if not os.path.isdir(year_path): 
                continue
            
            # Traverse the DAQ Run directories (e.g., 5132)
            for daq_run in os.listdir(year_path):
                run_path = os.path.join(year_path, daq_run)
                if not os.path.isdir(run_path): 
                    continue
                
                # Find and process all .root files inside
                for filename in os.listdir(run_path):

                    if not filename.endswith(".root"): 
                        continue

                    if counter%25==0:
                        print(counter)
                    
                    filepath = os.path.join(run_path, filename)
                    # Pass the csv_writer into the function
               
                    process_file(filepath, csv_writer)
                    counter += 1
                    
    print(f"Finished processing! Results saved to {output_filename}")

if __name__ == "__main__":
    main()

