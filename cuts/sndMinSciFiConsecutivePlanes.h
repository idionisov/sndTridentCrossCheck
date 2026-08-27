#pragma once

#include "sndSciFiBaseCut.h"

#include "TChain.h"
#include "sndScifiHit.h"

namespace snd {
  namespace trident_cuts {
    class minSciFiConsecutivePlanes : public snd::trident_cuts::sciFiBaseCut {
    public :
      minSciFiConsecutivePlanes(TChain * ch);
      ~minSciFiConsecutivePlanes(){;}

      bool passCut();

    };

  }
}
