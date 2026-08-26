#pragma once

#include "sndMuFilterBaseCut.h"

namespace snd::analysis_cuts {

  class minMaxUSQDCCut : public MuFilterBaseCut {
  private:
    double qdc_threshold;

  public:
    minMaxUSQDCCut(double threshold, TChain * ch);
    ~minMaxUSQDCCut() { ; }

    bool passCut() override;
  };

}
