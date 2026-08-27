#pragma once

#include "sndSciFiBaseCut.h"
#include "TChain.h"

namespace snd {
  namespace trident_cuts {
    class maxSciFiSignalCut : public snd::trident_cuts::sciFiBaseCut {
    private :
      double signalThreshold;
    public :
      maxSciFiSignalCut(double threshold, TChain * ch);
      ~maxSciFiSignalCut(){;}

      bool passCut();
    };
  }
}
