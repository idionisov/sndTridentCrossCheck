#include "sndDStoSciFiQDCRatioCut.h"

#include <vector>

#include "TClonesArray.h"
#include "TChain.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"

namespace snd::analysis_cuts {

  TClonesArray * DStoSciFiQDCRatioCut::scifiDigiHitCollection = 0;
  TClonesArray * DStoSciFiQDCRatioCut::muFilterDigiHitCollection = 0;

  DStoSciFiQDCRatioCut::DStoSciFiQDCRatioCut(double threshold, TChain * ch) : ratio_threshold(threshold) {
    if (scifiDigiHitCollection == 0) {
      scifiDigiHitCollection = new TClonesArray("sndScifiHit", 3000);
      ch->SetBranchAddress("Digi_ScifiHits", &scifiDigiHitCollection);
    }
    if (muFilterDigiHitCollection == 0) {
      muFilterDigiHitCollection = new TClonesArray("MuFilterHit", 470);
      ch->SetBranchAddress("Digi_MuFilterHits", &muFilterDigiHitCollection);
    }

    cutName = "DS/SciFi QDC Ratio >= " + std::to_string(ratio_threshold);
    shortName = "DStoSciFiQDCRatio";
    nbins = std::vector<int>{100};
    range_start = std::vector<double>{0.0};
    range_end = std::vector<double>{10.0};
    plot_var = std::vector<double>{-1.0};
  }

  bool DStoSciFiQDCRatioCut::passCut() {
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
        if (hit && hit->isValid() && hit->GetSystem() == 3) { // 3 = DS
          for (const auto& [key, value] : hit->GetAllSignals()) {
            totalDSQDC += value;
          }
        }
      }
    }

    double ratio = totalDSQDC / (totalSciFiQDC + 1e-4);
    plot_var[0] = ratio;

    if (ratio >= ratio_threshold) return true;
    return false;
  }

}
