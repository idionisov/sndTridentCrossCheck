#pragma once

#include "sndMuFilterBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace trident_cuts {
  
    class hasVetoHitsCut : public snd::trident_cuts::MuFilterBaseCut {
    public :
      hasVetoHitsCut(TChain * ch);
      ~hasVetoHitsCut(){;}
      bool passCut();
    };

  }
}
