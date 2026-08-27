#pragma once

#include "sndMuFilterBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace trident_cuts {

    class DSVetoCut : public snd::trident_cuts::MuFilterBaseCut {
    public :
      DSVetoCut(TChain * ch);
      ~DSVetoCut(){;}
      bool passCut();
    };

  }
}
