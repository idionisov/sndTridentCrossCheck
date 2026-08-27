#pragma once

#include "sndSciFiBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class maxPlaneSciFiSignalCut : public snd::trident_cuts::sciFiBaseCut {
    private :
      double planeThreshold;
    public :
      maxPlaneSciFiSignalCut(double threshold, TChain * ch);
      ~maxPlaneSciFiSignalCut(){;}

      bool passCut();
    };
  }
}
