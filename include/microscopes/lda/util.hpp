namespace util{

    void
    validate_probability_vector(const std::vector<float> &p){
        float sum = 0;
        for(auto x: p){
            assert(isfinite(x));
            assert(x >= 0);
            sum+=x;
        }
        assert(std::abs(1 - sum) < 0.01);
    }

    template<typename T> void
    removeFirst(std::vector<T> &v, T element){
        auto it = std::find(v.begin(),v.end(), element);
        if (it != v.end()) {
          v.erase(it);
        }
    }

    // http://stackoverflow.com/a/1267878/982745
    template< class T >
    std::vector<T>
    selectByIndex(const std::vector<T> &v, const std::vector<size_t> &index )  {
        std::vector<T> new_v;
        new_v.reserve(index.size());
        for(size_t i: index){
            new_v.push_back(v[i]);
        }

        return new_v;
    }


    template<class T>
    void
    normalize(std::vector<T> &v){
        for(auto x: v) {
            assert(isfinite(x));
        }
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(v.data(), v.size());
        vec /= vec.sum();
    }
}