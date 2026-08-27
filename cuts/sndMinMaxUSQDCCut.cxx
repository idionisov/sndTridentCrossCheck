#include "sndMinMaxUSQDCCut.h"

#include "TClonesArray.h"
#include "TChain.h"
#include "MuFilterHit.h"

#include <vector>

namespace snd::trident_cuts {

  minMaxUSQDCCut::minMaxUSQDCCut(double threshold, TChain * ch) : MuFilterBaseCut(ch), qdc_threshold(threshold) {
    cutName = "US Max Single Hit QDC >= " + std::to_string(qdc_threshold);

    shortName = "MinMaxUSQDC";
    nbins = std::vector<int>{100};
    range_start = std::vector<double>{0};
    range_end = std::vector<double>{200};
    plot_var = std::vector<double>{-1};
  }

  bool minMaxUSQDCCut::passCut() {
    MuFilterHit * hit;
    TIter hitIterator(muFilterDigiHitCollection);

    double max_qdc = 0.0;
    while ((hit = (MuFilterHit*) hitIterator.Next())) {
      if (hit->isValid() && hit->GetSystem() == 2) { // 2 = US
        double hit_qdc = 0.0;
        for (const auto& [key, value] : hit->GetAllSignals()) {
          hit_qdc += value;
        }
        if (hit_qdc > max_qdc) {
          max_qdc = hit_qdc;
        }
      }
    }
    plot_var[0] = max_qdc;
    if (max_qdc >= qdc_threshold) return true;
    return false;
  }
}
