#pragma once

#include <vector>

#include "sndBaseCut.h"

#include "TChain.h"
#include "TClonesArray.h"

namespace snd {
  namespace trident_cuts {
    class MuFilterBaseCut : public snd::trident_cuts::baseCut {

    protected :
      static TClonesArray * muFilterDigiHitCollection;

    public :
      MuFilterBaseCut(TChain * ch);
      virtual ~MuFilterBaseCut(){;}
      static void setupBranch(TChain * ch);
      static TClonesArray * getMuFilterDigiHitCollection() { return muFilterDigiHitCollection; }
    };

  }
}
