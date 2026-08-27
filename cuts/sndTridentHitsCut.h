#pragma once

#include "sndSciFiBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class tridentHitsCut : public snd::trident_cuts::sciFiBaseCut {
    private :
      int hitThreshold1;
      int hitThreshold2;
    public :
      tridentHitsCut(int threshold1, int threshold2, TChain * ch);
      ~tridentHitsCut(){;}

      bool passCut();
    };
  }
}
