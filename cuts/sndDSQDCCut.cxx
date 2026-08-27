#include "sndDSQDCCut.h"

#include "TClonesArray.h"
#include "TChain.h"
#include "MuFilterHit.h"

#include <vector>

namespace snd::trident_cuts {

  DSQDCCut::DSQDCCut(float threshold, TChain * ch) : MuFilterBaseCut(ch) {
    qdc_threshold = threshold;
    cutName = "Total DS QDC >= " + std::to_string(qdc_threshold);

    shortName = "DSQDC";
    nbins = std::vector<int>{100};
    range_start = std::vector<double>{0};
    range_end = std::vector<double>{5000};
    plot_var = std::vector<double>{-1};
  }

  bool DSQDCCut::passCut() {
    MuFilterHit * hit;
    TIter hitIterator(muFilterDigiHitCollection);

    float totQDC = 0.;
    while ((hit = (MuFilterHit*) hitIterator.Next())) {
      if (hit->isValid() && hit->GetSystem() == 3) { // 3 = DS
        for (const auto& [key, value] : hit->GetAllSignals()) {
          totQDC += value;
        }
      }
    }
    plot_var[0] = totQDC;
    if (totQDC >= qdc_threshold) return true;
    return false;
  }
}
