#pragma once

#include "sndMuFilterBaseCut.h"

#include "TChain.h"

namespace snd {
  namespace analysis_cuts {
  
    class USQDCCut : public snd::analysis_cuts::MuFilterBaseCut {
    private :
      float qdc_threshold;
    public :
      USQDCCut(float threshold, TChain * ch);
      ~USQDCCut(){;}
      bool passCut();
    };

  }
}
