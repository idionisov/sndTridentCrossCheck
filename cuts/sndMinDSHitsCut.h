#pragma once

#include "sndMuFilterBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace analysis_cuts {
    class minDSHitsCut : public snd::analysis_cuts::MuFilterBaseCut {
    private:
      int hit_threshold;
    public:
      minDSHitsCut(int threshold, TChain * ch);
      ~minDSHitsCut() {;}

      bool passCut() override;
    };
  }
}
