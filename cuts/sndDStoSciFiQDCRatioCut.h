#pragma once

#include "sndBaseCut.h"

#include "TChain.h"
#include "TClonesArray.h"
#include "sndScifiHit.h"
#include "MuFilterHit.h"

#include <vector>

namespace snd::analysis_cuts {

  class DStoSciFiQDCRatioCut : public baseCut {
  private:
    static TClonesArray * scifiDigiHitCollection;
    static TClonesArray * muFilterDigiHitCollection;
    double ratio_threshold;

  public:
    DStoSciFiQDCRatioCut(double threshold, TChain * ch);
    ~DStoSciFiQDCRatioCut() { ; }

    bool passCut() override;
  };

}
