#pragma once

#include "sndMuFilterBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class lastDSPlaneCut : public snd::trident_cuts::MuFilterBaseCut {
    private:
      int min_last_plane;
    public:
      lastDSPlaneCut(int min_plane, TChain * ch);
      ~lastDSPlaneCut() {;}

      bool passCut() override;
    };
  }
}
