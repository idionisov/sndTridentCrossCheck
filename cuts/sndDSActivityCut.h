#pragma once

#include "sndMuFilterBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace analysis_cuts {

    class DSActivityCut : public snd::analysis_cuts::MuFilterBaseCut {
    public :
      DSActivityCut(TChain * ch);
      ~DSActivityCut(){;}
      bool passCut();
    };

  }
}
