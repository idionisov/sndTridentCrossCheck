#pragma once

#include "sndSciFiBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class tridentDensityCut : public snd::trident_cuts::sciFiBaseCut {
    private :
      int radius;
      int threshold;
    public :
      tridentDensityCut(int r, int t, TChain * ch);
      ~tridentDensityCut(){;}

      bool passCut();
    };
  }
}
