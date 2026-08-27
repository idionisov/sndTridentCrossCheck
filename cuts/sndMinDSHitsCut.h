#pragma once

#include "sndMuFilterBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class minDSHitsCut : public snd::trident_cuts::MuFilterBaseCut {
    private:
      int hit_threshold;
    public:
      minDSHitsCut(int threshold, TChain * ch);
      ~minDSHitsCut() {;}

      bool passCut() override;
    };
  }
}
