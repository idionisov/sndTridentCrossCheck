#include "sndMinMaxSciFiSignalCut.h"

#include <vector>
#include <algorithm>

#include "TClonesArray.h"
#include "TChain.h"
#include "sndScifiHit.h"

namespace snd::trident_cuts {

  minMaxSciFiSignalCut::minMaxSciFiSignalCut(double threshold, TChain * ch) : sciFiBaseCut(ch), signal_threshold(threshold) {
    cutName = "SciFi Max Single Hit QDC >= " + std::to_string(signal_threshold);

    shortName = "MinMaxSciFiSignal";
    nbins = std::vector<int>{100};
    range_start = std::vector<double>{0};
    range_end = std::vector<double>{100};
    plot_var = std::vector<double>{-1};
  }

  bool minMaxSciFiSignalCut::passCut() {
    initializeEvent();

    double max_sig = 0.0;
    sndScifiHit * hit;
    TIter hitIterator(scifiDigiHitCollection);

    while ((hit = (sndScifiHit*) hitIterator.Next())) {
      if (hit && hit->isValid()) {
        double sig = hit->GetSignal();
        if (sig > max_sig) {
          max_sig = sig;
        }
      }
    }

    plot_var[0] = max_sig;
    if (max_sig >= signal_threshold) return true;
    return false;
  }
}
