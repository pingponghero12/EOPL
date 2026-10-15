#include <cstddef>
extern "C" void downsample(const double* a, double* b, size_t c, size_t h, size_t w, size_t f) {
    size_t lh=h/f, lw=w/f;
    for(size_t k=0;k<c;++k) for(size_t y=0;y<lh;++y) for(size_t x=0;x<lw;++x) {
        double sum=0;
        for(size_t dy=0;dy<f;++dy) for(size_t dx=0;dx<f;++dx)
            sum+=a[(k*h+y*f+dy)*w+x*f+dx];
        b[(k*lh+y)*lw+x]=sum/(f*f);
    }
}
extern "C" void scatter(const double* a, double* b, size_t c, size_t h, size_t w, double fraction) {
    for(size_t k=0;k<c;++k) for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x) {
        size_t base=k*h*w;
        b[base+y*w+x]=fraction*(a[base+y*w+x]+a[base+((y+1)%h)*w+x]
            +a[base+((y+h-1)%h)*w+x]+a[base+y*w+(x+1)%w]+a[base+y*w+(x+w-1)%w])/5;
    }
}
extern "C" void resample(const double* a, double* b, size_t c, size_t h, size_t w, size_t lh, size_t lw) {
    for(size_t k=0;k<c;++k) for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x)
        b[(k*h+y)*w+x]=a[(k*lh+y*lh/h)*lw+x*lw/w];
}
extern "C" void add(const double* a, const double* b, double* out, size_t n) {
    for(size_t i=0;i<n;++i) out[i]=a[i]+b[i];
}
