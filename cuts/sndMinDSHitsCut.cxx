#include "sndMinDSHitsCut.h"

#include "TClonesArray.h"
#include "TChain.h"
#include "MuFilterHit.h"

#include <vector>

namespace snd::analysis_cuts {

  minDSHitsCut::minDSHitsCut(int threshold, TChain * ch) : MuFilterBaseCut(ch) {
    hit_threshold = threshold;
    cutName = "DS hits >= " + std::to_string(hit_threshold);

    shortName = "MinDSHits";
    nbins = std::vector<int>{50};
    range_start = std::vector<double>{0};
    range_end = std::vector<double>{50};
    plot_var = std::vector<double>{-1};
  }

  bool minDSHitsCut::passCut() {
    MuFilterHit * hit;
    TIter hitIterator(muFilterDigiHitCollection);

    int n_ds_hits = 0;
    while ((hit = (MuFilterHit*) hitIterator.Next())) {
      if (hit->isValid() && hit->GetSystem() == 3) {
        n_ds_hits++;
      }
    }
    plot_var[0] = n_ds_hits;
    if (n_ds_hits >= hit_threshold) return true;
    return false;
  }
}
