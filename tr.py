import os
import glob
import ROOT
import SndlhcGeo
import SndlhcMuonReco
import SndlhcTracking

# --- Configuration ---
YEAR = 2023
DAQ_RUN = 6663
# Set to None or an empty list to process ALL events in the run
EVENTS = [6032227, 7029448, 12027270, 16657923, 21084195, 23081148, 29188092, 34100805, 38643259, 40530156, 54924576, 55758421, 56980139, 62996853, 64195812, 64683659, 65091790, 76371917, 78214307, 83393145, 101146792, 107216868, 108675029, 115168310, 115671435, 127734145, 130603272, 131522987, 136345931, 137876111, 145148917, 152759520, 155951413, 161114394, 161443282, 165329036, 169608593, 171173808, 172674754, 172737111, 176093204, 176556531, 184509065, 193049093, 195061777, 201166586, 206633992, 207582162, 211299525, 212652763, 226565859, 228735298, 229117881, 233159151, 233402118, 252198269, 255421050, 256293528, 263456498, 267708107, 271845389]

GEOFILE = f"/eos/experiment/sndlhc/convertedData/physics/{YEAR}/geofile_sndlhc_TI18_V3_{YEAR}.root"
RUN_DIR = f"/eos/experiment/sndlhc/convertedData/physics/{YEAR}/run_{DAQ_RUN:06d}"

# Find all files in the run directory
search_pattern = os.path.join(RUN_DIR, "sndsw_raw-*.root")
files = sorted(glob.glob(search_pattern))

if not files:
    print(f"Error: No files found in {RUN_DIR}")
    exit(1)

print(f"Found {len(files)} files to process in run {DAQ_RUN}.")

# --- Initialization ---
geo = SndlhcGeo.GeoInterface(GEOFILE)

run = ROOT.FairRunAna()
ioman = ROOT.FairRootManager.Instance()
ioman.SetTreeName("rawConv")

# Add all files to the source chain
source = ROOT.FairFileSource(files[0])
for i in range(1, len(files)):
    source.AddFile(files[i])
run.SetSource(source)

# Use TMemFile as a dummy sink to satisfy FairRoot
outFile = ROOT.TMemFile('dummy','CREATE')
sink = ROOT.FairRootFileSink(outFile)
run.SetSink(sink)

# Suppress some noise
xrdb = ROOT.FairRuntimeDb.instance()
xrdb.getContainer("FairBaseParSet").setStatic()
xrdb.getContainer("FairGeoParSet").setStatic()

# Add Hough Transform tracking task
ht_task = SndlhcMuonReco.MuonReco()
run.AddTask(ht_task)

ht_task.SetParFile(f"/afs/cern.ch/user/i/idioniso/snd_master/sndsw/trackingParams.xml")
ht_task.SetHoughSpaceFormat("linearSlopeIntercept")
ht_task.SetTrackingCase('passing_mu_Sf')

run.Init()

# Get the chain tree
tree = ioman.GetInTree()
if not tree:
    print("Error: Could not retrieve input tree.")
    exit(1)

# Initialize detector modules for the first event
tree.GetEvent(0)
geo.modules['Scifi'].InitEvent(tree.EventHeader)
geo.modules['MuFilter'].InitEvent(tree.EventHeader)

print(f"Starting loop over {tree.GetEntries()} total events...")

# --- Event Loop ---
for i_event, event in enumerate(tree):
    event_number = event.EventHeader.GetEventNumber()
    
    # Filter by event list if provided
    if EVENTS and event_number not in EVENTS:
        continue

    # Process tracking
    ht_task.kalman_tracks.Clear()
    ht_task.Exec(0)

    n_tracks = len(ht_task.kalman_tracks)
    if n_tracks == 3:
        print(f"Run {DAQ_RUN}, Event {event_number}: Trident found!")
    elif n_tracks == 2:
        print(f"Run {DAQ_RUN}, Event {event_number}: Two tracks found.")
    
    # Optional: progress indicator for large runs when not filtering
    if not EVENTS and i_event % 10000 == 0:
        print(f"Processed {i_event} events...")

print("Finished processing.")

