#pragma once

#include "sndMuFilterBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace analysis_cuts {

    class DSVetoCut : public snd::analysis_cuts::MuFilterBaseCut {
    public :
      DSVetoCut(TChain * ch);
      ~DSVetoCut(){;}
      bool passCut();
    };

  }
}
