#pragma once

#include "sndMuFilterBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace analysis_cuts {
    class lastDSPlaneCut : public snd::analysis_cuts::MuFilterBaseCut {
    private:
      int min_last_plane;
    public:
      lastDSPlaneCut(int min_plane, TChain * ch);
      ~lastDSPlaneCut() {;}

      bool passCut() override;
    };
  }
}
