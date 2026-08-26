#include "sndLastDSPlaneCut.h"

#include "TClonesArray.h"
#include "TChain.h"
#include "MuFilterHit.h"

#include <vector>

namespace snd::analysis_cuts {

  lastDSPlaneCut::lastDSPlaneCut(int min_plane, TChain * ch) : MuFilterBaseCut(ch) {
    min_last_plane = min_plane; // min_plane = 2 for 3rd DS plane (0: DS1, 1: DS2, 2: DS3, 3: DS4)
    cutName = "Last active DS plane >= " + std::to_string(min_last_plane + 1);

    shortName = "LastDSPlane";
    nbins = std::vector<int>{6};
    range_start = std::vector<double>{-1};
    range_end = std::vector<double>{5};
    plot_var = std::vector<double>{-1};
  }

  bool lastDSPlaneCut::passCut() {
    MuFilterHit * hit;
    TIter hitIterator(muFilterDigiHitCollection);

    int max_plane = -1;
    while ((hit = (MuFilterHit*) hitIterator.Next())) {
      if (hit->isValid() && hit->GetSystem() == 3) {
        int plane = hit->GetPlane();
        if (plane > max_plane) {
          max_plane = plane;
        }
      }
    }
    plot_var[0] = max_plane;
    if (max_plane >= min_last_plane) return true;
    return false;
  }
}
