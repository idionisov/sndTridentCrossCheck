#pragma once

#include "sndSciFiBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class maxPlaneSciFiHitsCut : public snd::trident_cuts::sciFiBaseCut {
    private :
      int planeThreshold;
    public :
      maxPlaneSciFiHitsCut(int threshold, TChain * ch);
      ~maxPlaneSciFiHitsCut(){;}

      bool passCut();
    };
  }
}
