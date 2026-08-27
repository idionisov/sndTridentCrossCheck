#pragma once

#include "sndMuFilterBaseCut.h"

namespace snd::trident_cuts {

  class minUSHitsCut : public MuFilterBaseCut {
  private:
    int hit_threshold;

  public:
    minUSHitsCut(int threshold, TChain * ch);
    ~minUSHitsCut() { ; }

    bool passCut() override;
  };

}
