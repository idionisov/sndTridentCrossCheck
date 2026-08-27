#pragma once

#include "sndMuFilterBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace trident_cuts {
  
    class vetoCut : public snd::trident_cuts::MuFilterBaseCut {
    public :
      vetoCut(TChain * ch);
      ~vetoCut(){;}
      bool passCut();
    };

  }
}
