#pragma once

#include "sndMuFilterBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace analysis_cuts {
  
    class vetoCut : public snd::analysis_cuts::MuFilterBaseCut {
    public :
      vetoCut(TChain * ch);
      ~vetoCut(){;}
      bool passCut();
    };

  }
}
