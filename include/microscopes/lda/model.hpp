#pragma once

#include <microscopes/models/base.hpp>
#include <microscopes/common/entity_state.hpp>
#include <microscopes/common/group_manager.hpp>
#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/common/util.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/io/schema.pb.h>
#include <distributions/special.hpp>

#include <cmath>
#include <vector>
#include <set>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <stdexcept>


namespace microscopes {
namespace lda {
namespace detail {

typedef std::vector<std::shared_ptr<models::group>> group_type;

template <template <typename> class GroupManager>
class state {
public:
	typedef typename GroupManager<group_type>::message_type message_type;

	state(const std::vector<std::shared_ptr<models::hypers>> &hypers,
		  const GroupManager<group_type> &groups)
	: hypers_(hypers), groups_(groups)
	{

	}

	int temp(){
		return 1;
	}

protected:
	std::vector<std::shared_ptr<models::hypers>> hypers_;
	GroupManager<group_type> groups_;
};

}

class model_definition {
public:
	model_definition(
      size_t n,
      const std::vector<std::shared_ptr<models::model>> &models)
    : n_(n), models_(models)
	{
		// MICROSCOPES_DCHECK(n > 0, "no entities given");
		// MICROSCOPES_DCHECK(models.size() > 0, "no features given");
	}
private:
	size_t n_;
	std::vector<std::shared_ptr<models::model>> models_;
};

}
}