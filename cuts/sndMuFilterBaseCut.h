#pragma once

#include <vector>

#include "sndBaseCut.h"

#include "TChain.h"
#include "TClonesArray.h"

namespace snd {
  namespace analysis_cuts {
    class MuFilterBaseCut : public snd::analysis_cuts::baseCut {

    protected :
      static TClonesArray * muFilterDigiHitCollection;

      MuFilterBaseCut(TChain * ch);
      ~MuFilterBaseCut(){;}
    };

  }
}
