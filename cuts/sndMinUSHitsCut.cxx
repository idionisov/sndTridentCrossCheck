#include "sndMinUSHitsCut.h"

#include "TClonesArray.h"
#include "TChain.h"
#include "MuFilterHit.h"

#include <vector>

namespace snd::analysis_cuts {

  minUSHitsCut::minUSHitsCut(int threshold, TChain * ch) : MuFilterBaseCut(ch) {
    hit_threshold = threshold;
    cutName = "US hits >= " + std::to_string(hit_threshold);

    shortName = "MinUSHits";
    nbins = std::vector<int>{50};
    range_start = std::vector<double>{0};
    range_end = std::vector<double>{50};
    plot_var = std::vector<double>{-1};
  }

  bool minUSHitsCut::passCut() {
    MuFilterHit * hit;
    TIter hitIterator(muFilterDigiHitCollection);

    int n_us_hits = 0;
    while ((hit = (MuFilterHit*) hitIterator.Next())) {
      if (hit->isValid() && hit->GetSystem() == 2) { // 2 = US
        n_us_hits++;
      }
    }
    plot_var[0] = n_us_hits;
    if (n_us_hits >= hit_threshold) return true;
    return false;
  }
}
