#pragma once

#include "sndSciFiBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class maxSciFiHitsCut : public snd::trident_cuts::sciFiBaseCut {
    private :
      int hitThreshold;
    public :
      maxSciFiHitsCut(int threshold, TChain * ch);
      ~maxSciFiHitsCut(){;}

      bool passCut();
    };
  }
}
