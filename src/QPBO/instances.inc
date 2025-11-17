#include "QPBO.h"

#ifdef _MSC_VER
#pragma warning(disable: 4661)
#endif

// Instantiations

template class QPBO<int>;
template class QPBO<float>;
template class QPBO<double>;

// Note: Specializations of get_type_information are now in QPBO.cpp
// before first use to avoid "explicit specialization after instantiation" error


