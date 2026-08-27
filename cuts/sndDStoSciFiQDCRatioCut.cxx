#include "sndDStoSciFiQDCRatioCut.h"
#include "sndSciFiBaseCut.h"
#include "sndMuFilterBaseCut.h"

#include <vector>
#include <map>

#include "TClonesArray.h"
#include "TChain.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"

namespace snd::trident_cuts {

  DStoSciFiQDCRatioCut::DStoSciFiQDCRatioCut(double threshold, TChain * ch) : ratio_threshold(threshold) {
    snd::trident_cuts::sciFiBaseCut::setupBranch(ch);
    snd::trident_cuts::MuFilterBaseCut::setupBranch(ch);

    cutName = "DS/SciFi QDC Ratio >= " + std::to_string(ratio_threshold);
    shortName = "DStoSciFiQDCRatio";
    nbins = std::vector<int>{100};
    range_start = std::vector<double>{0.0};
    range_end = std::vector<double>{10.0};
    plot_var = std::vector<double>{-1.0};
  }

  bool DStoSciFiQDCRatioCut::passCut() {
    TClonesArray * scifiDigiHitCollection = sciFiBaseCut::getSciFiDigiHitCollection();
    TClonesArray * muFilterDigiHitCollection = MuFilterBaseCut::getMuFilterDigiHitCollection();

    double totalSciFiQDC = 0.0;
    if (scifiDigiHitCollection) {
      sndScifiHit * hit;
      TIter hitIterator(scifiDigiHitCollection);
      while ((hit = (sndScifiHit*) hitIterator.Next())) {
        if (hit && hit->isValid()) {
          totalSciFiQDC += hit->GetSignal();
        }
      }
    }

    double totalDSQDC = 0.0;
    if (muFilterDigiHitCollection) {
      MuFilterHit * hit;
      TIter hitIterator(muFilterDigiHitCollection);
      while ((hit = (MuFilterHit*) hitIterator.Next())) {
        if (hit && hit->isValid() && hit->GetSystem() == 3) {
          std::map<TString, Float_t> signals = hit->SumOfSignals();
          for (auto const& [key, val] : signals) {
            totalDSQDC += val;
          }
        }
      }
    }

    double ratio = 0.0;
    if (totalSciFiQDC > 0.0) {
      ratio = totalDSQDC / totalSciFiQDC;
    }

    plot_var[0] = ratio;
    if (ratio >= ratio_threshold) return true;
    return false;
  }

}
