#pragma once

#include "sndMuFilterBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class DSQDCCut : public snd::trident_cuts::MuFilterBaseCut {
    private:
      float qdc_threshold;
    public:
      DSQDCCut(float threshold, TChain * ch);
      ~DSQDCCut() {;}

      bool passCut() override;
    };
  }
}
