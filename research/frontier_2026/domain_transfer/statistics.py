"""Weighted stratified estimators; exact empirical fractional tail, approximate CIs.

Reuse: SciPy t/norm and the project's paired-cluster formulas. Whole roots,
including all families/methods, are resampled within each stratum. The sorted
tail prefix is only a computational shortcut: uncovered replicates use the
entire sorted sample. No bootstrap replicate is dropped or approximated.
"""
import numpy as np
from scipy.stats import t


def checked_weights(weights, size):
    w=np.asarray(weights,dtype=float)
    if w.shape!=(size,) or not np.isfinite(w).all() or np.any(w<=0) or not np.isclose(w.sum(),1,rtol=0,atol=2e-14):
        raise ValueError('Positive unit-sum stratum weights required')
    return w


def weighted_mean_interval(samples, weights, *, alpha=.05, hypotheses=89):
    arrays=[np.asarray(x,float) for x in samples]
    if not arrays or any(x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Independent finite paired root differences required in every stratum')
    if not 0<alpha<1 or type(hypotheses) is not int or hypotheses<1:
        raise ValueError('Invalid error family')
    w=checked_weights(weights,len(arrays))
    mean=float(sum(weight*x.mean() for weight,x in zip(w,arrays)))
    # np.var can return a positive roundoff residual for e.g. [.1,.1,.1].
    # Only literally constant samples are degenerate; no tolerance erases
    # genuine, even very small, observed variation.
    parts=np.array([weight**2*(0. if np.all(x==x[0]) else x.var(ddof=1))/len(x)
                    for weight,x in zip(w,arrays)])
    variance=float(parts.sum());se=float(np.sqrt(variance))
    if variance==0:
        df=None;one=two=0.
    else:
        denominator=sum(part**2/(len(x)-1) for part,x in zip(parts,arrays))
        df=float(variance**2/denominator)
        one=float(t.ppf(1-alpha/hypotheses,df))*se
        two=float(t.ppf(1-alpha/(2*hypotheses),df))*se
    return {'mean':mean,'standard_error':se,'welch_degrees_of_freedom':df,
            'clusters_by_stratum':[len(x) for x in arrays],'weights':w.tolist(),
            'upper_one_sided':mean+one,'two_sided':[mean-two,mean+two],
            'alpha':alpha,'multiplicity':hypotheses,
            'scope':'Independent stratum paired-root Welch-t approximation, not distribution-free coverage'}


def weighted_cvar(values, masses, *, alpha=.95):
    x,w=np.asarray(values,float).ravel(),np.asarray(masses,float).ravel()
    if (x.shape!=w.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(w<0) or not np.isclose(w.sum(),1,rtol=0,atol=2e-13) or not 0<=alpha<1):
        raise ValueError('Finite outcomes and nonnegative unit probability masses required')
    order=np.argsort(-x,kind='stable');x,w=x[order],w[order]
    tail=1-alpha;before=np.cumsum(w)-w
    used=np.minimum(w,np.maximum(tail-before,0))
    return float(np.dot(used,x)/tail)


def _tails_for_counts(values, root_ids, masses, counts, *, alpha, prefix_fraction, order=None):
    """Exact tail of each count-weighted distribution, sorting values once."""
    if order is None:order=np.argsort(-values,kind='stable')
    ordered,roots,prob=values[order],root_ids[order],masses[order]
    length=max(1,int(np.ceil(len(order)*prefix_fraction)))
    def evaluate(selected_counts, end):
        weights=selected_counts[:,roots[:end]]*prob[:end]
        cumulative=np.cumsum(weights,axis=1)
        used=np.minimum(weights,np.maximum((1-alpha)-(cumulative-weights),0))
        return np.sum(used*ordered[:end],axis=1)/(1-alpha),cumulative[:,-1]
    result,covered=evaluate(counts,length)
    need=covered<1-alpha
    if np.any(need):result[need]=evaluate(counts[need],len(order))[0]
    return result,int(need.sum())


def stratified_tail_bootstrap(losses, weights, *, alpha=.95, confidence_alpha=.05,
                              hypotheses=89, resamples=49999, seed=202610129002,
                              batch=64, prefix_fraction=.125, progress=None):
    """Return all method CVaRs under common independently stratified root counts.

    Input is stratum × root × family × method. All strata have equal root
    count in this fixed design. Methods share each resample, families travel
    with their root; stratum masses are unchanged.
    """
    x=np.asarray(losses,float)
    if x.ndim!=4 or min(x.shape)<1 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError('Complete finite S×C×F×M loss array required')
    if (not 0<=alpha<1 or not 0<confidence_alpha<1 or type(hypotheses) is not int or hypotheses<1
            or type(resamples) is not int or resamples<100 or type(batch) is not int or batch<1
            or not 0<prefix_fraction<=1):
        raise ValueError('Invalid bootstrap settings')
    s,c,f,m=x.shape;w=checked_weights(weights,s)
    flat=x.reshape(-1,m)
    roots=np.broadcast_to(np.arange(s*c).reshape(s,c,1),(s,c,f)).ravel()
    masses=np.broadcast_to((w/(c*f))[:,None,None],(s,c,f)).ravel()
    point=np.array([weighted_cvar(flat[:,j],masses,alpha=alpha) for j in range(m)])
    orders=[np.argsort(-flat[:,j],kind='stable') for j in range(m)]
    rng=np.random.default_rng(seed);distribution=np.empty((resamples,m));fallback=0
    for start in range(0,resamples,batch):
        take=min(batch,resamples-start)
        counts=rng.multinomial(c,np.full(c,1/c),size=(take,s)).reshape(take,s*c)
        for j in range(m):
            result,n=_tails_for_counts(flat[:,j],roots,masses,counts,alpha=alpha,prefix_fraction=prefix_fraction,order=orders[j])
            distribution[start:start+take,j]=result;fallback+=n
        if progress is not None:progress(min(start+batch,resamples),resamples)
    return {'point':point,'distribution':distribution,'full_tail_computations':fallback,
            'resamples':resamples,'seed':seed,'batch':batch,'prefix_fraction':prefix_fraction,
            'confidence_alpha':confidence_alpha,'multiplicity':hypotheses,'cvar_alpha':alpha}


def bootstrap_difference(result, candidate, baseline):
    point=float(result['point'][candidate]-result['point'][baseline])
    sample=result['distribution'][:,candidate]-result['distribution'][:,baseline]
    q=result['confidence_alpha']/result['multiplicity']
    return {'difference':point,'upper_one_sided':float(2*point-np.quantile(sample,q,method='linear')),
            'bootstrap_standard_error':float(np.std(sample,ddof=1)),
            'resamples':result['resamples'],'seed':result['seed'],'confidence_alpha':result['confidence_alpha'],
            'multiplicity':result['multiplicity'],'cvar_alpha':result['cvar_alpha'],
            'expected_extreme_quantile_draws':q*result['resamples'],
            'scope':'Basic independently stratified paired-root bootstrap, approximate coverage'}
