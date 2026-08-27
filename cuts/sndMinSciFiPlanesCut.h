#pragma once

#include "sndSciFiBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class minSciFiPlanesCut : public snd::trident_cuts::sciFiBaseCut {
    private :
      int minPlanesH;
      int minPlanesV;
    public :
      minSciFiPlanesCut(int minH, int minV, TChain * ch);
      ~minSciFiPlanesCut(){;}

      bool passCut();
    };
  }
}
