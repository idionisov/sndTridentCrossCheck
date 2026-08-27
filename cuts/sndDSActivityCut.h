#pragma once

#include "sndMuFilterBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace trident_cuts {

    class DSActivityCut : public snd::trident_cuts::MuFilterBaseCut {
    public :
      DSActivityCut(TChain * ch);
      ~DSActivityCut(){;}
      bool passCut();
    };

  }
}
