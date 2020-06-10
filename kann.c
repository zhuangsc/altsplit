#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <stdarg.h>
#include <mpi.h>
#include "kann.h"

int kann_verbose = 3;

/******************************************
 *** @@BASIC: fundamental KANN routines ***
 ******************************************/

static void kad_ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c)
{
    int i, j, k, l, n_var;
    float *x, *g, *c;
    n_var = kad_size_var(n, a);
    x = *_x = (float*)realloc(*_x, n_var * sizeof(float));
    g = *_g = (float*)realloc(*_g, n_var * sizeof(float));
    c = *_c = (float*)realloc(*_c, kad_size_const(n, a) * sizeof(float));
    memset(g, 0, n_var * sizeof(float));
    for (i = j = k = 0; i < n; ++i) {
        kad_node_t *v = a[i];
        if (kad_is_var(v)) {
            l = kad_len(v);
            memcpy(&x[j], v->x, l * sizeof(float));
            free(v->x);
            v->x = &x[j];
            v->g = &g[j];
            j += l;
        } else if (kad_is_const(v)) {
            l = kad_len(v);
            memcpy(&c[k], v->x, l * sizeof(float));
            free(v->x);
            v->x = &c[k];
            k += l;
        }
    }
}

static void kad_ext_sync(int n, kad_node_t **a, float *x, float *g, float *c)
{
    int i, j, k;
    for (i = j = k = 0; i < n; ++i) {
        kad_node_t *v = a[i];
        if (kad_is_var(v)) {
            v->x = &x[j];
            v->g = &g[j];
            j += kad_len(v);
        } else if (kad_is_const(v)) {
            v->x = &c[k];
            k += kad_len(v);
        }
    }
}

kann_t *kann_new(kad_node_t *cost, int n_rest, ...)
{
    kann_t *a;
    int i, n_roots = 1 + n_rest, has_pivot = 0, has_recur = 0;
    kad_node_t **roots;
    va_list ap;

    if (cost->n_d != 0) return 0;

    va_start(ap, n_rest);
    roots = (kad_node_t**)malloc((n_roots + 1) * sizeof(kad_node_t*));
    for (i = 0; i < n_rest; ++i)
        roots[i] = va_arg(ap, kad_node_t*);
    roots[i++] = cost;
    va_end(ap);

    cost->ext_flag |= KANN_F_COST;
    a = (kann_t*)calloc(1, sizeof(kann_t));
    a->v = kad_compile_array(&a->n, n_roots, roots);

    for (i = 0; i < a->n; ++i) {
        if (a->v[i]->pre) has_recur = 1;
        if (kad_is_pivot(a->v[i])) has_pivot = 1;
    }
    if (has_recur && !has_pivot) { /* an RNN that doesn't have a pivot; then add a pivot on top of cost and recompile */
        cost->ext_flag &= ~KANN_F_COST;
        roots[n_roots-1] = cost = kad_avg(1, &cost), cost->ext_flag |= KANN_F_COST;
        free(a->v);
        a->v = kad_compile_array(&a->n, n_roots, roots);
    }
    kad_ext_collate(a->n, a->v, &a->x, &a->g, &a->c);
    free(roots);
    return a;
}

kann_t *kann_clone(kann_t *a, int batch_size)
{
    kann_t *b;
    b = (kann_t*)calloc(1, sizeof(kann_t));
    b->n = a->n;
    b->v = kad_clone(a->n, a->v, batch_size);
    kad_ext_collate(b->n, b->v, &b->x, &b->g, &b->c);
    return b;
}

kann_t *kann_unroll_array(kann_t *a, int *len)
{
    kann_t *b;
    int n_pivots;
    n_pivots = kad_n_pivots(a->n, a->v);
    b = (kann_t*)calloc(1, sizeof(kann_t));
    b->x = a->x, b->g = a->g, b->c = a->c; /* these arrays are shared */
    b->v = kad_unroll(a->n, a->v, &b->n, len);
    return b;
}

kann_t *kann_unroll(kann_t *a, ...)
{
    kann_t *b;
    va_list ap;
    int i, n_pivots, *len;
    n_pivots = kad_n_pivots(a->n, a->v);
    len = (int*)calloc(n_pivots, sizeof(int));
    va_start(ap, a);
    for (i = 0; i < n_pivots; ++i) len[i] = va_arg(ap, int);
    va_end(ap);
    b = kann_unroll_array(a, len);
    free(len);
    return b;
}

void kann_delete_unrolled(kann_t *a)
{
    if (a && a->mt) kann_mt(a, 0, 0);
    if (a && a->v) kad_delete(a->n, a->v);
    free(a);
}

void kann_delete(kann_t *a)
{
    if (a == 0) return;
    free(a->x); free(a->g); free(a->c);
    kann_delete_unrolled(a);
}

static void kann_switch_core(kann_t *a, int is_train)
{
    int i;
    for (i = 0; i < a->n; ++i)
        if (a->v[i]->op == 12 && a->v[i]->n_child == 2)
            *(int32_t*)a->v[i]->ptr = !!is_train;
}

#define chk_flg(flag, mask) ((mask) == 0 || ((flag) & (mask)))
#define chk_lbl(label, query) ((query) == 0 || (label) == (query))

int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{
    int i, k, r = -1;
    for (i = k = 0; i < a->n; ++i)
        if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
            ++k, r = i;
    return k == 1? r : k == 0? -1 : -2;
}

int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x)
{
    int i, k;
    if (x == 0) return 0;
    for (i = k = 0; i < a->n; ++i)
        if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
            a->v[i]->x = x[k++];
    return k;
}

int kann_feed_dim(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{
    int i, k, n = 0;
    for (i = k = 0; i < a->n; ++i)
        if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
            ++k, n = a->v[i]->n_d > 1? kad_len(a->v[i]) / a->v[i]->d[0] : a->v[i]->n_d == 1? a->v[i]->d[0] : 1;
    return k == 1? n : k == 0? -1 : -2;
}

static float kann_cost_core(kann_t *a, int cost_label, int cal_grad)
{
    int i_cost;
    float cost;
    i_cost = kann_find(a, KANN_F_COST, cost_label);
    assert(i_cost >= 0);
    cost = *kad_eval_at(a->n, a->v, i_cost);
    if (cal_grad) kad_grad(a->n, a->v, i_cost);
    return cost;
}

int kann_eval(kann_t *a, uint32_t ext_flag, int ext_label)
{
    int i, k;
    for (i = k = 0; i < a->n; ++i)
        if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
            ++k, a->v[i]->tmp = 1;
    kad_eval_marked(a->n, a->v);
    return k;
}

void kann_rnn_start(kann_t *a)
{
    int i;
    kann_set_batch_size(a, 1);
    for (i = 0; i < a->n; ++i) {
        kad_node_t *p = a->v[i];
        if (p->pre) { /* NB: BE CAREFUL of the interaction between kann_rnn_start() and kann_set_batch_size() */
            kad_node_t *q = p->pre;
            if (q->x) memcpy(p->x, q->x, kad_len(p) * sizeof(float));
            else memset(p->x, 0, kad_len(p) * sizeof(float));
            if (q->n_child > 0) free(q->x);
            q->x = p->x;
        }
    }
}

void kann_rnn_end(kann_t *a)
{
    int i;
    kad_ext_sync(a->n, a->v, a->x, a->g, a->c);
    for (i = 0; i < a->n; ++i)
        if (a->v[i]->pre && a->v[i]->pre->n_child > 0)
            a->v[i]->pre->x = (float*)calloc(kad_len(a->v[i]->pre), sizeof(float));
}

static int kann_class_error_core(const kann_t *ann, int *base)
{
    int i, j, k, m, n, off, n_err = 0;
    for (i = 0, *base = 0; i < ann->n; ++i) {
        kad_node_t *p = ann->v[i];
        if ((p->op == 13 || p->op == 22) && p->n_child == 2 && p->n_d == 0) { /* ce_bin or ce_multi */
            kad_node_t *x = p->child[0], *t = p->child[1];
            n = t->d[t->n_d - 1], m = kad_len(t) / n;
            for (j = off = 0; j < m; ++j, off += n) {
                float t_sum = 0.0f, t_min = 1.0f, t_max = 0.0f, x_max = 0.0f, x_min = 1.0f;
                int x_max_k = -1, t_max_k = -1;
                for (k = 0; k < n; ++k) {
                    float xk = x->x[off+k], tk = t->x[off+k];
                    t_sum += tk;
                    t_min = t_min < tk? t_min : tk;
                    x_min = x_min < xk? x_min : xk;
                    if (t_max < tk) t_max = tk, t_max_k = k;
                    if (x_max < xk) x_max = xk, x_max_k = k;
                }
                if (t_sum - 1.0f == 0 && t_min >= 0.0f && x_min >= 0.0f && x_max <= 1.0f) {
                    ++(*base);
                    n_err += (x_max_k != t_max_k);
                }
            }
        }
    }
    return n_err;
}

/*************************
 * @@MT: multi-threading *
 *************************/

#ifdef HAVE_PTHREAD
#include <pthread.h>

struct mtaux_t;

typedef struct { /* per-worker data */
    kann_t *a;
    float cost;
    int action;
    pthread_t tid;
    struct mtaux_t *g;
} mtaux1_t;

typedef struct mtaux_t { /* cross-worker data */
    int n_threads, max_batch_size;
    int cal_grad, cost_label;
    volatile int n_idle; /* we will be busy waiting on this, so volatile necessary */
    pthread_mutex_t mtx;
    pthread_cond_t cv;
    mtaux1_t *mt;
} mtaux_t;

static void *mt_worker(void *data) /* pthread worker */
{
    mtaux1_t *mt1 = (mtaux1_t*)data;
    mtaux_t *mt = mt1->g;
    for (;;) {
        int action;
        pthread_mutex_lock(&mt->mtx);
        mt1->action = 0;
        ++mt->n_idle;
        while (mt1->action == 0)
            pthread_cond_wait(&mt->cv, &mt->mtx);
        action = mt1->action;
        pthread_mutex_unlock(&mt->mtx);
        if (action == -1) break;

        mt1->cost = kann_cost_core(mt1->a, mt->cost_label, mt->cal_grad);
    }
    pthread_exit(0);
}

static void mt_destroy(mtaux_t *mt) /* de-allocate an entire mtaux_t struct */
{
    int i;
    pthread_mutex_lock(&mt->mtx);
    mt->n_idle = 0;
    for (i = 1; i < mt->n_threads; ++i) mt->mt[i].action = -1;
    pthread_cond_broadcast(&mt->cv);
    pthread_mutex_unlock(&mt->mtx);
    for (i = 1; i < mt->n_threads; ++i) pthread_join(mt->mt[i].tid, 0);
    for (i = 0; i < mt->n_threads; ++i) kann_delete(mt->mt[i].a);
    free(mt->mt);
    pthread_cond_destroy(&mt->cv);
    pthread_mutex_destroy(&mt->mtx);
    free(mt);
}

void kann_mt(kann_t *ann, int n_threads, int max_batch_size)
{
    mtaux_t *mt;
    int i, k;

    if (n_threads <= 1) {
        if (ann->mt) mt_destroy((mtaux_t*)ann->mt);
        ann->mt = 0;
        return;
    }
    if (n_threads > max_batch_size) n_threads = max_batch_size;
    if (n_threads <= 1) return;

    mt = (mtaux_t*)calloc(1, sizeof(mtaux_t));
    mt->n_threads = n_threads, mt->max_batch_size = max_batch_size;
    pthread_mutex_init(&mt->mtx, 0);
    pthread_cond_init(&mt->cv, 0);
    mt->mt = (mtaux1_t*)calloc(n_threads, sizeof(mtaux1_t));
    for (i = k = 0; i < n_threads; ++i) {
        int size = (max_batch_size - k) / (n_threads - i);
        mt->mt[i].a = kann_clone(ann, size);
        mt->mt[i].g = mt;
        k += size;
    }
    for (i = 1; i < n_threads; ++i)
        pthread_create(&mt->mt[i].tid, 0, mt_worker, &mt->mt[i]);
    while (mt->n_idle < n_threads - 1); /* busy waiting until all threads in sync */
    ann->mt = mt;
}

float kann_cost(kann_t *a, int cost_label, int cal_grad)
{
    mtaux_t *mt = (mtaux_t*)a->mt;
    int i, j, B, k, n_var;
    float cost;

    if (mt == 0) return kann_cost_core(a, cost_label, cal_grad);
    B = kad_sync_dim(a->n, a->v, -1); /* get the current batch size */
    assert(B <= mt->max_batch_size); /* TODO: can be relaxed */
    n_var = kann_size_var(a);

    pthread_mutex_lock(&mt->mtx);
    mt->cost_label = cost_label, mt->cal_grad = cal_grad;
    for (i = k = 0; i < mt->n_threads; ++i) {
        int size = (B - k) / (mt->n_threads - i);
        for (j = 0; j < a->n; ++j)
            if (kad_is_feed(a->v[j]))
                mt->mt[i].a->v[j]->x = &a->v[j]->x[k * kad_len(a->v[j]) / a->v[j]->d[0]];
        kad_sync_dim(mt->mt[i].a->n, mt->mt[i].a->v, size); /* TODO: we can point ->x to internal nodes, too */
        k += size;
        memcpy(mt->mt[i].a->x, a->x, n_var * sizeof(float));
        mt->mt[i].action = 1;
    }
    mt->n_idle = 0;
    pthread_cond_broadcast(&mt->cv);
    pthread_mutex_unlock(&mt->mtx);

    mt->mt[0].cost = kann_cost_core(mt->mt[0].a, cost_label, cal_grad);
    while (mt->n_idle < mt->n_threads - 1); /* busy waiting until all threads in sync */

    memset(a->g, 0, n_var * sizeof(float));
    for (i = k = 0, cost = 0.0f; i < mt->n_threads; ++i) {
        int size = (B - k) / (mt->n_threads - i);
        cost += mt->mt[i].cost * size / B;
        kad_saxpy(n_var, (float)size / B, mt->mt[i].a->g, a->g);
        k += size;
    }
    for (j = 0; j < a->n; ++j) { /* copy values back at recurrent nodes (needed by textgen; TODO: temporary solution) */
        kad_node_t *p = a->v[j];
        if (p->pre && p->n_d >= 2 && p->d[0] == B) {
            for (i = k = 0; i < mt->n_threads; ++i) {
                kad_node_t *q = mt->mt[i].a->v[j];
                memcpy(&p->x[k], q->x, kad_len(q) * sizeof(float));
                k += kad_len(q);
            }
        }
    }
    return cost;
}

int kann_class_error(const kann_t *ann, int *base)
{
    mtaux_t *mt = (mtaux_t*)ann->mt;
    int i, n_err = 0, b = 0;
    if (mt == 0) return kann_class_error_core(ann, base);
    for (i = 0; i < mt->n_threads; ++i) {
        n_err += kann_class_error_core(mt->mt[i].a, &b);
        *base += b;
    }
    return n_err;
}

void kann_switch(kann_t *ann, int is_train)
{
    mtaux_t *mt = (mtaux_t*)ann->mt;
    int i;
    if (mt == 0) {
        kann_switch_core(ann, is_train);
        return;
    }
    for (i = 0; i < mt->n_threads; ++i)
        kann_switch_core(mt->mt[i].a, is_train);
}
#else
void kann_mt(kann_t *ann, int n_threads, int max_batch_size) {}
float kann_cost(kann_t *a, int cost_label, int cal_grad) { return kann_cost_core(a, cost_label, cal_grad); }
int kann_class_error(const kann_t *a, int *base) { return kann_class_error_core(a, base); }
void kann_switch(kann_t *ann, int is_train) { return kann_switch_core(ann, is_train); }
#endif

/***********************
 *** @@IO: model I/O ***
 ***********************/

#define KANN_MAGIC "KAN\1"

void kann_save_fp(FILE *fp, kann_t *ann)
{
    kann_set_batch_size(ann, 1);
    fwrite(KANN_MAGIC, 1, 4, fp);
    kad_save(fp, ann->n, ann->v);
    fwrite(ann->x, sizeof(float), kann_size_var(ann), fp);
    fwrite(ann->c, sizeof(float), kann_size_const(ann), fp);
}

void kann_save(const char *fn, kann_t *ann)
{
    FILE *fp;
    fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
    kann_save_fp(fp, ann);
    fclose(fp);
}

kann_t *kann_load_fp(FILE *fp)
{
    char magic[4];
    kann_t *ann;
    int n_var, n_const;

    fread(magic, 1, 4, fp);
    if (strncmp(magic, KANN_MAGIC, 4) != 0) {
        fclose(fp);
        return 0;
    }
    ann = (kann_t*)calloc(1, sizeof(kann_t));
    ann->v = kad_load(fp, &ann->n);
    n_var = kad_size_var(ann->n, ann->v);
    n_const = kad_size_const(ann->n, ann->v);
    ann->x = (float*)malloc(n_var * sizeof(float));
    ann->g = (float*)calloc(n_var, sizeof(float));
    ann->c = (float*)malloc(n_const * sizeof(float));
    fread(ann->x, sizeof(float), n_var, fp);
    fread(ann->c, sizeof(float), n_const, fp);
    kad_ext_sync(ann->n, ann->v, ann->x, ann->g, ann->c);
    return ann;
}

kann_t *kann_load(const char *fn)
{
    FILE *fp;
    kann_t *ann;
    fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
    ann = kann_load_fp(fp);
    fclose(fp);
    return ann;
}

/**********************************************
 *** @@LAYER: layers and model generation ***
 **********************************************/

/********** General but more complex APIs **********/

kad_node_t *kann_new_leaf_array(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, int32_t d[KAD_MAX_DIM])
{
    int i, len, off = offset && par? *offset : -1;
    kad_node_t *p;

    if (off >= 0 && par[off]) return par[(*offset)++];
    p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
    p->n_d = n_d, p->flag = flag;
    memcpy(p->d, d, n_d * sizeof(int32_t));
    len = kad_len(p);
    p->x = (float*)calloc(len, sizeof(float));
    if (p->n_d <= 1) {
        for (i = 0; i < len; ++i)
            p->x[i] = x0_01;
    } else {
        double sdev_inv;
        //sdev_inv = 1.0 / sqrt((double)len / p->d[0]);
        sdev_inv = 2.0 / sqrt((double)len / p->d[0] + p->d[0]);
        for (i = 0; i < len; ++i)
            p->x[i] = (float)(kad_drand_normal(0) * sdev_inv);
    }
    if (off >= 0) par[off] = p, ++(*offset);
    return p;
}

kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, ...)
{
    int32_t i, d[KAD_MAX_DIM];
    va_list ap;
    va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
    return kann_new_leaf_array(offset, par, flag, x0_01, n_d, d);
}

kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *in, int n1)
{
    int n0;
    kad_node_t *w, *b;
    n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
    w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
    b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
    memset(b->x, 0, kad_len(b)*sizeof(float));
    printf("Dense %d -> %d\n", n0, n1);
    return kad_add(kad_cmul(in, w), b);
}

kad_node_t *kann_layer_dense2_nobias(int *offset, kad_node_p *par, kad_node_t *in, int n1)
{
    int n0;
    kad_node_t *w, *b;
    n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
    w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
    printf("NB Dense %d -> %d\n", n0, n1);
    return kad_cmul(in, w);
}

static void kann_print_weights(kad_node_t *p) 
{
#ifdef __DRYRUN__
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //if (kad_len(p)*size > 40) return;
    float *global = NULL;
    if (rank == 0) {
        global = calloc(kad_len(p)*size, sizeof(float));
    }
    MPI_Gather(p->x, kad_len(p), MPI_FLOAT, global, kad_len(p), MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        qsort(global, kad_len(p)*size, sizeof(float), kad_compare_float);
        for (int i = 0; i < kad_len(p)*size; i++)
            fprintf(stdout, "%lf\n", global[i]);
    }
    //fprintf(stdout, "------------------------------RANK: %d\n", rank);
    //for (int i = 0; i < kad_len(p); i++) {
    //    fprintf(stdout, "%lf", p->x[i]);
    //}
    //printf("\n");
    //MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void kann_drand_init(int seed) 
{
    drand_t.cnt = 0;
    srand48(seed);
}

int kann_drand_fetch(double *val)
{
    *val = drand48();
    drand_t.cnt += 1;
    return drand_t.cnt;
}

int kann_drand_count()
{
    return drand_t.cnt;
}

static double kann_drand_normal()
{
    double fac, rsq, v1, v2;
    do {
        kann_drand_fetch(&v1);
        kann_drand_fetch(&v2);
        v1 = 2.0 * v1 - 1.0;
        v2 = 2.0 * v2 - 1.0;
        rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    return v2 * fac;
}

static int skip_random(int n)
{
    float a = 0.0;
    int i;
    for (i = 0; i < n; i++)
        a += (float) kann_drand_normal();
    return i;
}

static void reassign_weights_on_ranking(int split_mode, kad_node_t *p, int n0, int n1, int print)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double sdev_inv;
    int len;
    len = kad_len(p);
    sdev_inv = 1.0 / sqrt((double)len / p->d[0]);

    switch (split_mode) {
        case CNN_REDUCE:
            len *= size;
            sdev_inv = 1.0 / sqrt((double)len / p->d[0]/size);
            for (int r = 0; r < size; r++) {
                if (rank == r) {
                    for (int cl = 0; cl < n1; cl++)
                        for (int pl = 0; pl < n0; pl++)
                            p->x[cl*n0+pl] = (float)(kann_drand_normal() * sdev_inv);
                } else {
                    skip_random(n0*n1);
                }
            }
            break;
        case NOREDUCE:
            for (int r = 0; r < size; r++) {
                if (rank == r) {
                    for (int cl = 0; cl < n1; cl++)
                        for (int pl = 0; pl < n0; pl++)
                            p->x[cl*n0+pl] = (float)(kann_drand_normal() * sdev_inv);
                } else {
                    skip_random(n0*n1);
                }
            }
            break;
        case ALLREDUCE:
            for (int r = 0; r < size; r++) {
                if (rank == r) {
                    for (int cl = 0; cl < n1; cl++)
                        for (int pl = 0; pl < n0*size; pl++)
                            p->x[cl*n0*size+pl] = (float)(kann_drand_normal() * sdev_inv);
                } else {
                    skip_random(n0*n1*size);
                }
            }
            break;
        case ALTREDUCE:
        case ALTREDUCE_ONEWAY:
            len *= size;
            sdev_inv = 1.0 / sqrt((double)len / p->d[0]);
            for (int cl = 0; cl < n1; cl++) {
                for (int r = 0; r < size; r++) {
                    if (rank == r) {
                        for (int pl = 0; pl < n0; pl++) {
                            p->x[cl*n0+pl] = (float)(kann_drand_normal() * sdev_inv);
                        }
                    } else {
                        skip_random(n0);
                    }
                }
            }
            break;
        case ALLREDUCE_ONEWAY:
            len *= size;
            sdev_inv = 1.0 / sqrt((double)len / p->d[0]);
            for (int cl = 0; cl < n1; cl++) {
                for (int r = 0; r < size; r++) {
                    if (rank == r) {
                        for (int pl = 0; pl < n0; pl++) {
                            p->x[cl*n0+pl] = (float)(kann_drand_normal() * sdev_inv);
                        }
                    } else {
                        skip_random(n0);
                    }
                }
            }
            break;
        default:
            break;
    }
    if (print)
        kann_print_weights(p);
}

kad_node_t *kann_layer_dense2_mpi(int *offset, kad_node_p *par, kad_node_t *in, int n1, int split_mode)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n0;
    kad_node_t *w, *b;
    n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
    switch (split_mode) {
        case ALLREDUCE:
            w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0*size);
            reassign_weights_on_ranking(split_mode, w, n0, n1, 0);
            break;
        case ALLREDUCE_ONEWAY:
            w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
            reassign_weights_on_ranking(split_mode, w, n0, n1, 0);
            break;
        default:
            w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
            reassign_weights_on_ranking(split_mode, w, n0, n1, 0);
            break;
    }
   
    b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
    //reassign_bias_on_ranking(split_mode, b, n1); //TODO add the function
    //printf("rank: %d cnt: %d\n", rank, kann_drand_count());

    if (split_mode == NOREDUCE) {
        return kad_add(kad_cmul(in, w), b);
    } 
    else if (split_mode == ALTREDUCE) {
        return kad_add(kad_cmul_altsplit(in, w), b);
    }
    else if (split_mode == ALLREDUCE) {
        return kad_add(kad_cmul_allsplit(in, w), b);
    }
    else if (split_mode == ALTREDUCE_ONEWAY) {
        return kad_add(kad_cmul_altsplit_out(in, w), b);
    }
    else if (split_mode == ALLREDUCE_ONEWAY) {
        return kad_add(kad_cmul_allsplit_out(in, w), b);
    }
    else 
        return 0;
}

kad_node_t *kann_layer_dropout2(int *offset, kad_node_p *par, kad_node_t *t, float r)
{
    kad_node_t *x[2], *cr;
    cr = kann_new_leaf2(offset, par, KAD_CONST, r, 0);
    x[0] = t, x[1] = kad_dropout(t, cr);
    return kad_switch(2, x);
}

kad_node_t *kann_layer_layernorm2(int *offset, kad_node_t **par, kad_node_t *in)
{
    int n0;
    kad_node_t *alpha, *beta;
    n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
    alpha = kann_new_leaf2(offset, par, KAD_VAR, 1.0f, 1, n0);
    beta  = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n0);
    return kad_add(kad_mul(kad_stdnorm(in), alpha), beta);
}

static inline kad_node_t *cmul_norm2(int *offset, kad_node_t **par, kad_node_t *x, kad_node_t *w, int use_norm)
{
    return use_norm? kann_layer_layernorm2(offset, par, kad_cmul(x, w)) : kad_cmul(x, w);
}

kad_node_t *kann_layer_rnn2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag)
{
    int n0, n1 = h0->d[h0->n_d-1], use_norm = !!(rnn_flag & KANN_RNN_NORM);
    kad_node_t *t, *w, *u, *b, *out;

    u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
    b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
    t = cmul_norm2(offset, par, h0, u, use_norm);
    if (in) {
        n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
        w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
        t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
    }
    out = kad_tanh(kad_add(t, b));
    out->pre = h0;
    return out;
}

kad_node_t *kann_layer_gru2(int *offset, kad_node_t **par, kad_node_t *in, kad_node_t *h0, int rnn_flag)
{
    int n0 = 0, n1 = h0->d[h0->n_d-1], use_norm = !!(rnn_flag & KANN_RNN_NORM);
    kad_node_t *t, *r, *z, *w, *u, *b, *s, *out;

    if (in) n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
    /* z = sigm(x_t * W_z + h_{t-1} * U_z + b_z) */
    u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
    b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
    t = cmul_norm2(offset, par, h0, u, use_norm);
    if (in) {
        w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
        t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
    }
    z = kad_sigm(kad_add(t, b));
    /* r = sigm(x_t * W_r + h_{t-1} * U_r + b_r) */
    u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
    b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
    t = cmul_norm2(offset, par, h0, u, use_norm);
    if (in) {
        w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
        t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
    }
    r = kad_sigm(kad_add(t, b));
    /* s = tanh(x_t * W_s + (h_{t-1} # r) * U_s + b_s) */
    u = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n1);
    b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
    t = cmul_norm2(offset, par, kad_mul(r, h0), u, use_norm);
    if (in) {
        w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
        t = kad_add(cmul_norm2(offset, par, in, w, use_norm), t);
    }
    s = kad_tanh(kad_add(t, b));
    /* h_t = z # h_{t-1} + (1 - z) # s */
    out = kad_add(kad_mul(kad_1minus(z), s), kad_mul(z, h0));
    out->pre = h0;
    return out;
}

/********** APIs without offset & par **********/

kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, int n_d, ...)
{
    int32_t i, d[KAD_MAX_DIM];
    va_list ap;
    va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
    return kann_new_leaf_array(0, 0, flag, x0_01, n_d, d);
}

kad_node_t *kann_new_scalar(uint8_t flag, float x) { return kann_new_leaf(flag, x, 0); }
kad_node_t *kann_new_weight(int n_row, int n_col) { return kann_new_leaf(KAD_VAR, 0.0f, 2, n_row, n_col); }
kad_node_t *kann_new_vec(int n, float x) { return kann_new_leaf(KAD_VAR, x, 1, n); }
kad_node_t *kann_new_bias(int n) { return kann_new_vec(n, 0.0f); }
kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col) { return kann_new_leaf(KAD_VAR, 0.0f, 4, n_out, n_in, k_row, k_col); }
kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len) { return kann_new_leaf(KAD_VAR, 0.0f, 3, n_out, n_in, kernel_len); }

kad_node_t *kann_layer_input(int n1)
{
    kad_node_t *t;
    t = kad_feed(2, 1, n1), t->ext_flag |= KANN_F_IN;
    return t;
}

kad_node_t *kann_layer_dense(kad_node_t *in, int n1) { return kann_layer_dense2(0, 0, in, n1); }
kad_node_t *kann_layer_dense_nobias(kad_node_t *in, int n1) { return kann_layer_dense2_nobias(0, 0, in, n1); }
kad_node_t *kann_layer_dense_mpi(kad_node_t *in, int n1, int split_mode) { return kann_layer_dense2_mpi(0, 0, in, n1, split_mode); }
kad_node_t *kann_layer_dropout(kad_node_t *t, float r) { return kann_layer_dropout2(0, 0, t, r); }
kad_node_t *kann_layer_layernorm(kad_node_t *in) { return kann_layer_layernorm2(0, 0, in); }

kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, int rnn_flag)
{
    kad_node_t *h0;
    h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
    h0->x = (float*)calloc(n1, sizeof(float));
    return kann_layer_rnn2(0, 0, in, h0, rnn_flag);
}

kad_node_t *kann_layer_gru(kad_node_t *in, int n1, int rnn_flag)
{
    kad_node_t *h0;
    h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
    h0->x = (float*)calloc(n1, sizeof(float));
    return kann_layer_gru2(0, 0, in, h0, rnn_flag);
}

static kad_node_t *kann_cmul_norm(kad_node_t *x, kad_node_t *w)
{
    return kann_layer_layernorm(kad_cmul(x, w));
}

kad_node_t *kann_layer_lstm(kad_node_t *in, int n1, int rnn_flag)
{
    int n0;
    kad_node_t *i, *f, *o, *g, *w, *u, *b, *h0, *c0, *c, *out;
    kad_node_t *(*cmul)(kad_node_t*, kad_node_t*) = (rnn_flag & KANN_RNN_NORM)? kann_cmul_norm : kad_cmul;

    n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
    h0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
    h0->x = (float*)calloc(n1, sizeof(float));
    c0 = (rnn_flag & KANN_RNN_VAR_H0)? kad_var(0, 0, 2, 1, n1) : kad_const(0, 2, 1, n1);
    c0->x = (float*)calloc(n1, sizeof(float));

    /* i = sigm(x_t * W_i + h_{t-1} * U_i + b_i) */
    w = kann_new_weight(n1, n0);
    u = kann_new_weight(n1, n1);
    b = kann_new_bias(n1);
    i = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
    /* f = sigm(x_t * W_f + h_{t-1} * U_f + b_f) */
    w = kann_new_weight(n1, n0);
    u = kann_new_weight(n1, n1);
    b = kann_new_vec(n1, 1.0f); /* see Jozefowicz et al on using a large bias */
    f = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
    /* o = sigm(x_t * W_o + h_{t-1} * U_o + b_o) */
    w = kann_new_weight(n1, n0);
    u = kann_new_weight(n1, n1);
    b = kann_new_bias(n1);
    o = kad_sigm(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
    /* g = tanh(x_t * W_g + h_{t-1} * U_g + b_g) */
    w = kann_new_weight(n1, n0);
    u = kann_new_weight(n1, n1);
    b = kann_new_bias(n1);
    g = kad_tanh(kad_add(kad_add(cmul(in, w), cmul(h0, u)), b));
    /* c_t = c_{t-1} # f + g # i */
    c = kad_add(kad_mul(f, c0), kad_mul(g, i)); /* can't be kad_mul(c0, f)!!! */
    c->pre = c0;
    /* h_t = tanh(c_t) # o */
    if (rnn_flag & KANN_RNN_NORM) c = kann_layer_layernorm(c); /* see Ba et al (2016) about how to apply layer normalization to LSTM */
    out = kad_mul(kad_tanh(c), o);
    out->pre = h0;
    return out;
}

kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c)
{
    kad_node_t *w;
    w = kann_new_weight_conv2d(n_flt, in->d[1], k_rows, k_cols);
    return kad_conv2d(in, w, stride_r, stride_c, pad_r, pad_c);
}

kad_node_t *kann_layer_conv2d_mpi(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c)
{
    kad_node_t *w;
    w = kann_new_weight_conv2d(n_flt, in->d[1], k_rows, k_cols);
    reassign_weights_on_ranking(CNN_REDUCE, w, n_flt, in->d[1]*k_rows*k_cols, 0);
    //printf("%d %d %d %d\n", n_flt, in->d[1], k_rows, k_cols);
    //printf("cnt: %d\n", kann_drand_count());
    return kad_conv2d(in, w, stride_r, stride_c, pad_r, pad_c);
}

kad_node_t *kann_layer_conv1d(kad_node_t *in, int n_flt, int k_size, int stride, int pad)
{
    kad_node_t *w;
    w = kann_new_weight_conv1d(n_flt, in->d[1], k_size);
    return kad_conv1d(in, w, stride, pad);
}

kad_node_t *kann_layer_cost(kad_node_t *t, int n_out, int cost_type)
{
    kad_node_t *cost = 0, *truth = 0;
    assert(cost_type == KANN_C_CEB || cost_type == KANN_C_CEM || cost_type == KANN_C_CEB_NEG || cost_type == KANN_C_MSE);
    t = kann_layer_dense(t, n_out);
    truth = kad_feed(2, 1, n_out), truth->ext_flag |= KANN_F_TRUTH;
    if (cost_type == KANN_C_MSE) {
        cost = kad_mse(t, truth);
    } else if (cost_type == KANN_C_CEB) {
        t = kad_sigm(t);
        cost = kad_ce_bin(t, truth);
    } else if (cost_type == KANN_C_CEB_NEG) {
        t = kad_tanh(t);
        cost = kad_ce_bin_neg(t, truth);
    } else if (cost_type == KANN_C_CEM) {
        t = kad_softmax(t);
        cost = kad_ce_multi(t, truth);
    }
    t->ext_flag |= KANN_F_OUT, cost->ext_flag |= KANN_F_COST;
    return cost;
}

kad_node_t *kann_layer_cost_mpi(kad_node_t *t, int n_out, int cost_type, int split_mode)
{
    kad_node_t *cost = 0, *truth = 0;
    assert(cost_type == KANN_C_CEB || cost_type == KANN_C_CEM || cost_type == KANN_C_CEB_NEG || cost_type == KANN_C_MSE);
    t = kann_layer_dense_mpi(t, n_out, split_mode);
    truth = kad_feed(2, 1, n_out), truth->ext_flag |= KANN_F_TRUTH;
    if (cost_type == KANN_C_MSE) {
        cost = kad_mse(t, truth);
    } else if (cost_type == KANN_C_CEB) {
        t = kad_sigm(t);
        cost = kad_ce_bin(t, truth);
    } else if (cost_type == KANN_C_CEB_NEG) {
        t = kad_tanh(t);
        cost = kad_ce_bin_neg(t, truth);
    } else if (cost_type == KANN_C_CEM) {
        t = kad_softmax(t);
        cost = kad_ce_multi(t, truth);
    }
    t->ext_flag |= KANN_F_OUT, cost->ext_flag |= KANN_F_COST;
    return cost;
}

void kann_shuffle(int n, int *s)
{
    int i, j, t;
    double jtmp;
    for (i = 0; i < n; ++i) s[i] = i;
    for (i = n; i > 0; --i) {
        //j = (int)(i * kad_drand(0));
        //j = (int)(i * drand48());
        kann_drand_fetch(&jtmp);
        j = (int)(i * jtmp);
        t = s[j], s[j] = s[i-1], s[i-1] = t;
    }
}

/***************************
 *** @@MIN: minimization ***
 ***************************/

#ifdef __SSE__
#include <xmmintrin.h>

void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
    int i, n4 = n>>2<<2;
    __m128 vh, vg, vr, vt, vd, vd1, tmp, vtiny;
    vh = _mm_set1_ps(h0);
    vd = _mm_set1_ps(decay);
    vd1 = _mm_set1_ps(1.0f - decay);
    vtiny = _mm_set1_ps(1e-6f);
    for (i = 0; i < n4; i += 4) {
        vt = _mm_loadu_ps(&t[i]);
        vr = _mm_loadu_ps(&r[i]);
        vg = _mm_loadu_ps(&g[i]);
        if (h) vh = _mm_loadu_ps(&h[i]);
        vr = _mm_add_ps(_mm_mul_ps(vd1, _mm_mul_ps(vg, vg)), _mm_mul_ps(vd, vr));
        _mm_storeu_ps(&r[i], vr);
        tmp = _mm_sub_ps(vt, _mm_mul_ps(_mm_mul_ps(vh, _mm_rsqrt_ps(_mm_add_ps(vtiny, vr))), vg));
        _mm_storeu_ps(&t[i], tmp);
    }
    for (; i < n; ++i) {
        r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
        t[i] -= (h? h[i] : h0) / sqrtf(1e-6f + r[i]) * g[i];
    }
}
#else
void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
    int i;
    for (i = 0; i < n; ++i) {
        float lr = h? h[i] : h0;
        r[i] = (1.0f - decay) * g[i] * g[i] + decay * r[i];
        t[i] -= lr / sqrtf(1e-6f + r[i]) * g[i];
    }
}
#endif

float kann_grad_clip(float thres, int n, float *g)
{
    int i;
    double s2 = 0.0;
    for (i = 0; i < n; ++i)
        s2 += g[i] * g[i];
    s2 = sqrt(s2);
    if (s2 > thres)
        for (i = 0, s2 = 1.0 / s2; i < n; ++i)
            g[i] *= (float)s2;
    return (float)s2 / thres;
}

/****************************************************************
 *** @@XY: simpler API for network with a single input/output ***
 ****************************************************************/

int kann_train_fnn1(kann_t *ann, float lr, int mini_size, int max_epoch, int max_drop_streak, float frac_val, int n, float **_x, float **_y)
{
    int i, j, *shuf, n_train, n_val, n_in, n_out, n_var, n_const, drop_streak = 0, min_set = 0;
    float **x, **y, *x1, *y1, *r, min_val_cost = FLT_MAX, *min_x, *min_c;

    n_in = kann_dim_in(ann);
    n_out = kann_dim_out(ann);
    if (n_in < 0 || n_out < 0) return -1;
    n_var = kann_size_var(ann);
    n_const = kann_size_const(ann);
    r = (float*)calloc(n_var, sizeof(float));
    shuf = (int*)malloc(n * sizeof(int));
    x = (float**)malloc(n * sizeof(float*));
    y = (float**)malloc(n * sizeof(float*));
    kann_shuffle(n, shuf);
    for (j = 0; j < n; ++j)
        x[j] = _x[shuf[j]], y[j] = _y[shuf[j]];
    n_val = (int)(n * frac_val);
    n_train = n - n_val;
    min_x = (float*)malloc(n_var * sizeof(float));
    min_c = (float*)malloc(n_const * sizeof(float));

    x1 = (float*)malloc(n_in  * mini_size * sizeof(float));
    y1 = (float*)malloc(n_out * mini_size * sizeof(float));
    kann_feed_bind(ann, KANN_F_IN,    0, &x1);
    kann_feed_bind(ann, KANN_F_TRUTH, 0, &y1);

    for (i = 0; i < max_epoch; ++i) {
        int n_proc = 0, n_train_err = 0, n_val_err = 0, n_train_base = 0, n_val_base = 0;
        double train_cost = 0.0, val_cost = 0.0;
        kann_shuffle(n_train, shuf);
        kann_switch(ann, 1);
        while (n_proc < n_train) {
            int b, c, ms = n_train - n_proc < mini_size? n_train - n_proc : mini_size;
            for (b = 0; b < ms; ++b) {
                memcpy(&x1[b*n_in],  x[shuf[n_proc+b]], n_in  * sizeof(float));
                memcpy(&y1[b*n_out], y[shuf[n_proc+b]], n_out * sizeof(float));
            }
            kann_set_batch_size(ann, ms);
            train_cost += kann_cost(ann, 0, 1) * ms;
            c = kann_class_error(ann, &b);
            n_train_err += c, n_train_base += b;
            kann_RMSprop(n_var, lr, 0, 0.9f, ann->g, ann->x, r);
            n_proc += ms;
        }
        train_cost /= n_train;
        kann_switch(ann, 0);
        n_proc = 0;
        while (n_proc < n_val) {
            int b, c, ms = n_val - n_proc < mini_size? n_val - n_proc : mini_size;
            for (b = 0; b < ms; ++b) {
                memcpy(&x1[b*n_in],  x[n_train+n_proc+b], n_in  * sizeof(float));
                memcpy(&y1[b*n_out], y[n_train+n_proc+b], n_out * sizeof(float));
            }
            kann_set_batch_size(ann, ms);
            val_cost += kann_cost(ann, 0, 0) * ms;
            c = kann_class_error(ann, &b);
            n_val_err += c, n_val_base += b;
            n_proc += ms;
        }
        if (n_val > 0) val_cost /= n_val;
        if (kann_verbose >= 3) {
            fprintf(stderr, "epoch: %d; training cost: %g", i+1, train_cost);
            if (n_train_base) fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_train_err / n_train);
            if (n_val > 0) {
                fprintf(stderr, "; validation cost: %g", val_cost);
                if (n_val_base) fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_val_err / n_val);
            }
            fputc('\n', stderr);
        }
        if (i >= max_drop_streak && n_val > 0) {
            if (val_cost < min_val_cost) {
                min_set = 1;
                memcpy(min_x, ann->x, n_var * sizeof(float));
                memcpy(min_c, ann->c, n_const * sizeof(float));
                drop_streak = 0;
                min_val_cost = (float)val_cost;
            } else if (++drop_streak >= max_drop_streak)
                lr /= 10;
                //break;
        }
    }
    if (min_set) {
        memcpy(ann->x, min_x, n_var * sizeof(float));
        memcpy(ann->c, min_c, n_const * sizeof(float));
    }

    free(min_c); free(min_x); free(y1); free(x1); free(y); free(x); free(shuf); free(r);
    return i;
}

static void shuffle_sum(int n, int *shuf) 
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0)  return;
    int tot = 0;
    for (int i = 0; i < n; i++)
        tot += shuf[i];
    printf("shuffle %d\n", tot);
}

void kann_allreduce_insert(kann_t *ann, int bs) 
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = ann->n;
    kad_node_t **a = ann->v;
    for (int i = 0; i < n; ++i)
        if (a[i]->op == 38) {
            a[i]->allval = calloc(kad_len(a[i]->child[0])*size*bs, sizeof(float));
            a[i]->allgrad = calloc(kad_len(a[i]->child[0])*size*bs, sizeof(float));
        } 
}

void kann_allreduce_check(kann_t *ann) 
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = ann->n;
    kad_node_t **a = ann->v;
    for (int i = 0; i < n; ++i)
        if (a[i]->op == 38)
            printf("rank %d allval %p allgrad %p\n", rank, a[i]->allval, a[i]->allgrad);
}

void kann_allreduce_free(kann_t *ann)
{
    int n = ann->n;
    kad_node_t **a = ann->v;
    for (int i = 0; i < n; ++i) {
        if (a[i]->op == 38) {
            free(a[i]->allval);
            free(a[i]->allgrad);
        }
    }
}

kann_t* kann_insert_alt_mlp_mpi(kad_node_t *t, int n_out, int loss_type, int n_h_layers, int tot_neurons, float h_dropout, int size, int bs, int rank)
{
    int n_h_neurons = tot_neurons / size;
    
    if (!n_h_neurons) {
        if (rank == 0)
            fprintf(stderr, "Neurons %d not sufficient for the MPI processes %d\n", tot_neurons, size);
        return 0;
    }

    int i, k;

    // Hidden layers
    for (i = 0, k = 0; i < n_h_layers; ++i) {
        k = (i % 2 == 0);
        if (k) {  // Distributed
            t = kad_relu(kann_layer_dense_mpi(t, n_h_neurons, NOREDUCE));
            //t = kann_layer_dropout(kad_relu(kann_layer_dense_mpi(t, n_h_neurons, NOREDUCE)), h_dropout);
        } else {
            t = kad_relu(kann_layer_dense_mpi(t, tot_neurons, ALTREDUCE));
            //t = kann_layer_dropout(kad_relu(kann_layer_dense_mpi(t, tot_neurons, ALTREDUCE)), h_dropout);
        }
    }

    // Output layer
    /* The preceeding layer is a distributed layer, output layer needs:
     * 1) Reassign weights
     * 2) Allreduce propagating forward
     * 3) No reduction propagating backward
     */
    if (k) {
        return kann_new(kann_layer_cost_mpi(t, n_out, loss_type, ALTREDUCE_ONEWAY), 0);
    } 
    /* The preceeding layer is a duplicated layer, output layer needs:
     * 1) The same weights across MPI processes
     * 2) No reduction propagating forward
     * 3) No reduction propagating backward
     */
    else { 
        fprintf(stderr, "Warning the layer preceeding the output layer being a duplicated layer is not supported currently\n");
        return 0;
        //return kann_new(kann_layer_cost(t, n_out, loss_type), 0);
    }
}

kann_t* kann_insert_all_mlp_mpi(kad_node_t *t, int n_out, int loss_type, int n_h_layers, int tot_neurons, float h_dropout, int size, int bs, int rank)
{
    int n_h_neurons = tot_neurons / size;
    
    if (!n_h_neurons) {
        if (rank == 0)
            fprintf(stderr, "Neurons %d not sufficient for the MPI processes %d\n", tot_neurons, size);
        return 0;
    }

    // Hidden layers
    t = kad_relu(kann_layer_dense_mpi(t, n_h_neurons, NOREDUCE));
    //t = kann_layer_dropout(kad_relu(kann_layer_dense_mpi(t, n_h_neurons, NOREDUCE)), h_dropout);
    for (int i = 1; i < n_h_layers; ++i) {
        t = kad_relu(kann_layer_dense_mpi(t, n_h_neurons, ALLREDUCE));
        //t = kann_layer_dropout(kad_relu(kann_layer_dense_mpi(t, n_h_neurons, ALLREDUCE)), h_dropout);
    }
    kann_t *a = kann_new(kann_layer_cost_mpi(t, n_out, loss_type, ALLREDUCE_ONEWAY), 0);
    kann_allreduce_insert(a, bs);
    return a;
}

int kann_train_fnn1_mpi(kann_t *ann, float lr, int mini_size, int max_epoch, int max_drop_streak, float frac_val, int n, float **_x, float **_y, int rank, int batches)
{
    int i, j, *shuf, n_train, n_val, n_in, n_out, n_var, n_const, drop_streak = 0, min_set = 0;
    float **x, **y, *x1, *y1, *r, min_val_cost = FLT_MAX, *min_x, *min_c;

    n_in = kann_dim_in(ann);
    n_out = kann_dim_out(ann);
    if (n_in < 0 || n_out < 0) return -1;
    n_var = kann_size_var(ann);
    n_const = kann_size_const(ann);
    r = (float*)calloc(n_var, sizeof(float));
    shuf = (int*)malloc(n * sizeof(int));
    x = (float**)malloc(n * sizeof(float*));
    y = (float**)malloc(n * sizeof(float*));
    kann_shuffle(n, shuf);
    for (j = 0; j < n; ++j)
        x[j] = _x[shuf[j]], y[j] = _y[shuf[j]];
    n_val = (int)(n * frac_val);
    n_train = n - n_val;
    min_x = (float*)malloc(n_var * sizeof(float));
    min_c = (float*)malloc(n_const * sizeof(float));

    x1 = (float*)malloc(n_in  * mini_size * sizeof(float));
    y1 = (float*)malloc(n_out * mini_size * sizeof(float));
    kann_feed_bind(ann, KANN_F_IN,    0, &x1);
    kann_feed_bind(ann, KANN_F_TRUTH, 0, &y1);

    if (batches) {
        double start_time[batches];
        double training_time[batches];
        double update_time[batches];
        double batch_time[batches];
        int mini_batch = 0;

        int n_proc = 0, n_train_err = 0, n_val_err = 0, n_train_base = 0, n_val_base = 0;
        double train_cost = 0.0, val_cost = 0.0;
        kann_shuffle(n_train, shuf);
        kann_switch(ann, 1);
        fprintf(stdout, "Batches start\n");
        while (n_proc < n_train && mini_batch < batches) {
            int b, c, ms = n_train - n_proc < mini_size? n_train - n_proc : mini_size;
            for (b = 0; b < ms; ++b) {
                memcpy(&x1[b*n_in],  x[shuf[n_proc+b]], n_in  * sizeof(float));
                memcpy(&y1[b*n_out], y[shuf[n_proc+b]], n_out * sizeof(float));
            }
            kann_set_batch_size(ann, ms);
            start_time[mini_batch] = MPI_Wtime();
            train_cost += kann_cost(ann, 0, 1) * ms;
            training_time[mini_batch] = MPI_Wtime();
            kann_RMSprop(n_var, lr, 0, 0.9f, ann->g, ann->x, r);
            update_time[mini_batch] = MPI_Wtime();
            c = kann_class_error(ann, &b);
            n_train_err += c, n_train_base += b;
            batch_time[mini_batch] = MPI_Wtime();
            n_proc += ms;
            fprintf(stdout, "Training %d Elapse %f\n", mini_batch, batch_time[mini_batch]-start_time[mini_batch]);
            mini_batch += 1;
        }

        if (rank == 0) {
            double training_tot_time = 0.0; double training_avg_time = 0.0;
            double update_tot_time = 0.0; double update_avg_time = 0.0;
            double class_tot_time = 0.0; double class_avg_time = 0.0;
            double batch_tot_time = 0.0; double batch_avg_time = 0.0;
            for (int i = 1; i < batches; i++) {
                training_tot_time += training_time[i] - start_time[i];
                update_tot_time += update_time[i] - training_time[i];
                batch_tot_time += batch_time[i]  - start_time[i];
            }
            training_avg_time = training_tot_time / batches;
            update_avg_time = update_tot_time / batches;
            batch_avg_time = batch_tot_time / batches;
            fprintf(stdout, "Training-elapse %d tot %f avg %f\n", mini_batch, training_tot_time, training_avg_time);
            fprintf(stdout, "Update-elapse %d tot %f avg %f\n", mini_batch, update_tot_time, update_avg_time);
            fprintf(stdout, "Batch-elapse %d tot %f avg %f\n", mini_batch, batch_tot_time, batch_avg_time);
        }
        return mini_batch;
    }

    double start_time;
    double epoch_time;
    for (i = 0; i < max_epoch; ++i) {
        int n_proc = 0, n_train_err = 0, n_val_err = 0, n_train_base = 0, n_val_base = 0;
        double train_cost = 0.0, val_cost = 0.0;
        kann_shuffle(n_train, shuf);
        kann_switch(ann, 1);
        start_time = MPI_Wtime();
        while (n_proc < n_train) {
            int b, c, ms = n_train - n_proc < mini_size? n_train - n_proc : mini_size;
            for (b = 0; b < ms; ++b) {
                memcpy(&x1[b*n_in],  x[shuf[n_proc+b]], n_in  * sizeof(float));
                memcpy(&y1[b*n_out], y[shuf[n_proc+b]], n_out * sizeof(float));
            }
            kann_set_batch_size(ann, ms);
            train_cost += kann_cost(ann, 0, 1) * ms;
            kann_RMSprop(n_var, lr, 0, 0.9f, ann->g, ann->x, r);
            c = kann_class_error(ann, &b);
            n_train_err += c, n_train_base += b;
            n_proc += ms;
        }
        train_cost /= n_train;
        kann_switch(ann, 0);
        n_proc = 0;
        while (n_proc < n_val) {
            int b, c, ms = n_val - n_proc < mini_size? n_val - n_proc : mini_size;
            for (b = 0; b < ms; ++b) {
                memcpy(&x1[b*n_in],  x[n_train+n_proc+b], n_in  * sizeof(float));
                memcpy(&y1[b*n_out], y[n_train+n_proc+b], n_out * sizeof(float));
            }
            kann_set_batch_size(ann, ms);
            val_cost += kann_cost(ann, 0, 0) * ms;
            c = kann_class_error(ann, &b);
            n_val_err += c, n_val_base += b;
            n_proc += ms;
        }
        epoch_time = MPI_Wtime() - start_time;

        if (n_val > 0) val_cost /= n_val;
        if (kann_verbose >= 3 && rank == 0) {
            fprintf(stderr, "epoch: %d; training cost: %g", i+1, train_cost);
            if (n_train_base) fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_train_err / n_train);
            if (n_val > 0) {
                fprintf(stderr, "; validation cost: %g", val_cost);
                if (n_val_base) fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_val_err / n_val);
            }
            fprintf(stderr, " elapse time: %f\n", epoch_time);
            fputc('\n', stderr);
        }
        if (i >= max_drop_streak && n_val > 0) {
            if (val_cost < min_val_cost) {
                min_set = 1;
                memcpy(min_x, ann->x, n_var * sizeof(float));
                memcpy(min_c, ann->c, n_const * sizeof(float));
                drop_streak = 0;
                min_val_cost = (float)val_cost;
            } else if (++drop_streak >= max_drop_streak)
                lr /= 10;
                //break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (min_set) {
        memcpy(ann->x, min_x, n_var * sizeof(float));
        memcpy(ann->c, min_c, n_const * sizeof(float));
    }

    free(min_c); free(min_x); free(y1); free(x1); free(y); free(x); free(shuf); free(r);
    return i;
}

static void _kann_gen_epoch(kann_dok_t *dok, int max_user_id, int max_item_id, int n, int num_neg, float **x, float **x0, float *y0, int *shuf)
{
    int n_ext = n + num_neg * max_user_id;
    int id, cnt;

    /* Append negative interactions */
    for(int i = n, id = 0, cnt = 0; i <= n_ext; i++, cnt++) {
        if (x[i] == NULL) {
            //FIXME Need to free it at the end
            float *xy = calloc(2, sizeof(float));
            x[i] = xy;
            x[i][0] = (float) id;
        }
        while(1) {
            int val = lrand48() % max_item_id;
            int hit = 1;
            for(int j = 0; j < dok->sze[id]; j++) {
                if (val == dok->idx[id][j]) {
                    hit = 0;
                    break;
                }
            }
            if (hit) {
                x[i][1] = (float) val;
                break;
            }
        }
        //fprintf(stderr, "id %.2f cnt %d val %.2f\n", x[i][0], cnt, x[i][1]);
        if (cnt == num_neg) {
            id += 1;
            cnt = 0;
        }
    }

    kann_shuffle(n_ext, shuf);

    /* Fill x0 and y0 */
    for (int j = 0; j < n_ext; ++j) {
        x0[j] = x[shuf[j]];
        if (shuf[j] < n)
            y0[j] = (float) 1;
        else
            y0[j] = (float) 0;
    }
}

static inline void _kann_gen_batch(int max_user_id, int max_item_id, float *x1, float *y1, float **x0, float *y0, int n_in, int n_out, int n_proc, float *tmp_oh, int ms)
{
    for (int b = 0; b < ms; ++b) {
        tmp_oh[(int)x0[n_proc+b][0]] = (float) 1;
        tmp_oh[(int)x0[n_proc+b][1]+max_user_id] = (float) 1;
        memcpy(&x1[b*n_in], tmp_oh, n_in * sizeof(float));
        tmp_oh[(int)x0[n_proc+b][0]] = (float) 0;
        tmp_oh[(int)x0[n_proc+b][1]+max_user_id] = (float) 0;
        y1[b*n_out] = y0[n_proc+b];
    }
#ifdef __VERIFY__
    /* Assert that the inputs are in ont-hot format*/
    float sum = (float) 0;
    for (int j = 0; j < n_in*ms; j++)
        sum += x1[j];
    assert(sum - 2.0 * ms == 0);
#endif
}

static inline void _kann_gen_validation(int idx, int n_val_col, float **_val, float *x1, float *y1, float *tmp_oh, int n_in, int max_user_id)
{
    float *val = _val[idx];
    for(int i = 0; i < n_val_col; i++) {
        tmp_oh[idx] = (float) 1;
        tmp_oh[(int)val[i]+max_user_id] = (float) 1;
        memcpy(&x1[i*n_in], tmp_oh, n_in * sizeof(float));
        tmp_oh[idx] = (float) 0;
        tmp_oh[(int)val[i]+max_user_id] = (float) 0;
    }
#ifdef __VERIFY__
    //for(int i = 0; i < n_val_col; i++) 
    //    fprintf(stderr, "%d ", (int)val[i]);
    //fprintf(stderr, "\n---------------------------\n");

    /* Assert that the inputs are in ont-hot format*/
    float sum = (float) 0;
    for (int j = 0; j < n_in*n_val_col; j++)
        sum += x1[j];
    assert(sum - 2.0 * n_val_col == 0);
#endif
}

static inline void _kann_evaluate_recommendation(float *hits, float *ndcgs, int n_val_row, float t_cost)
{
    float hits_sum = (float) 0;
    float ndcg_sum = (float) 0;
    for (int i = 0; i < n_val_row; i++) {
        hits_sum += hits[i];
        ndcg_sum += ndcgs[i];
    }
    memset(hits, 0, n_val_row * sizeof(float));
    memset(ndcgs, 0, n_val_row * sizeof(float));
    fprintf(stdout, "HR: %f, NDCG: %f, loss: %f ", hits_sum/n_val_row, ndcg_sum/n_val_row, t_cost);
}

static inline void _kann_recommender_error(const kann_t *ann, float *hit, float *ndcg, int n_val_col, int top_K, float *score, int *score_id, float *tmp)
{
    int i, j, k, m, n, off, n_err = 0;
    for (i = 0; i < ann->n; ++i) {
        kad_node_t *p = ann->v[i];
        if ((p->op == 13 || p->op == 22) && p->n_child == 2 && p->n_d == 0) { /* ce_bin or ce_multi */
            kad_node_t *x = p->child[0], *t = p->child[1];
            n = t->d[t->n_d - 1], m = kad_len(t) / n;
#ifdef __VERIFY__
            assert(n_val_col == m && n == 1);
#endif
            /* Fill scores*/
            memcpy(score, x->x, m * sizeof(float));
            memcpy(tmp, score, m * sizeof(float));
            /* Sort socres for top-K*/
            for (int top = 0; top < top_K; top++) {
                float top_score = FLT_MIN;
                for (j = 0; j < n_val_col; j++ ) {
                    if (score[j] - top_score > 0) {
                        top_score = score[j];
                        score_id[top] = j;
                    }
                }
                score[score_id[top]] = FLT_MIN;
            }
            for (j = 0; j < top_K; j++) {
                if (score_id[j] == 0) {
                    /* Hit Ratio */
                    *hit = (float) 1;
                    /* NDCG */
                    *ndcg = logf(2) / logf(j+2);
                }
            }
#ifdef __VERIFY__
            //for (j = 0; j < top_K; j++) 
            //    printf("%d ", score_id[j]);
            //printf("\n");
            //for (j = 0; j < top_K; j++) 
            //    printf("%f ", tmp[score_id[j]]);
            //printf("\n");

            //if (*hit == 1) {
            //    int sum = 0;
            //    for (j = 0; j < top_K; j++)
            //        sum += score_id[0];
            //    assert(sum == 0);
            //}
#endif
        }
    }
}

int kann_train_fnn1_mpi_onehot(kann_t *ann, float lr, int mini_size, int max_epoch, int max_drop_streak, 
    float frac_val, int n, float **_x, int n_val_row, int n_val_col, float **_val, int rank, int batches, 
    int max_user_id, int max_item_id, kann_dok_t *dok, int num_neg, int top_K)
{
    int i, j, *shuf, n_train, n_val, n_in, n_out, n_var, n_const, drop_streak = 0, min_set = 0;
    float **x, **y, *x1, *y1, *r, min_val_cost = FLT_MAX, *min_x, *min_c;
    int n_ext = n + num_neg * max_user_id;

    n_in = kann_dim_in(ann);
    n_out = kann_dim_out(ann);
    if (n_in < 0 || n_out < 0) return -1;
    n_var = kann_size_var(ann);
    n_const = kann_size_const(ann);
    r = (float*)calloc(n_var, sizeof(float));
    shuf = (int*)malloc(n * sizeof(int));
    x = (float**)malloc(n_ext * sizeof(float*));
    y = (float**)malloc(n_ext * sizeof(float*));
    //kann_shuffle(n, shuf);
    memcpy(x, _x, n * sizeof(float*));
    //for (j = 0; j < n; ++j)
    //    x[j] = _x[j];
        //x[j] = _x[shuf[j]];//, y[j] = _y[shuf[j]];
    n_val = (int)(n_ext * frac_val);
    n_train = n_ext; //- n_val;
    min_x = (float*)malloc(n_var * sizeof(float));
    min_c = (float*)malloc(n_const * sizeof(float));

    int bs = mini_size > n_val_col ? mini_size : n_val_col;
    x1 = (float*)malloc(n_in  * bs * sizeof(float));
    y1 = (float*)malloc(n_out * bs * sizeof(float));
    kann_feed_bind(ann, KANN_F_IN,    0, &x1);
    kann_feed_bind(ann, KANN_F_TRUTH, 0, &y1);

    int *shuf_ext = (int*) malloc(n_ext * sizeof(int));
    float **x0    = (float**) calloc(n_ext, sizeof(float*));
    float *y0     = (float*) calloc(n_ext, sizeof(float));
    float *tmp_oh = (float*) calloc(max_user_id+max_item_id, sizeof(float));
    float *score  = (float*) calloc(n_val_col, sizeof(float));
    int *score_id = (int*) calloc(top_K, sizeof(int));
    float *hits  = (float*) calloc(n_val_row, sizeof(float));
    float *ndcgs = (float*) calloc(n_val_row, sizeof(float));

    double start_time, epoch_time, training_time, eval_time;
    fprintf(stderr, "n_in %d, n_out %d, n %d, max_user_id %d, max_item_id %d, n_val_row %d, n_val_col %d\n", n_in, n_out, n, max_user_id, max_item_id, n_val_row, n_val_col);
    /* Initial Evaluation */
    kann_switch(ann, 0);
    float tmp[n_val_col];
    start_time = MPI_Wtime();
    for (int i = 0; i < n_val_row; i++) {
        _kann_gen_validation(i, n_val_col, _val, x1, y1, tmp_oh, n_in, max_user_id);
        kann_set_batch_size(ann, n_val_col);
        kann_cost(ann, 0, 0);
        _kann_recommender_error(ann, &hits[i], &ndcgs[i], n_val_col, top_K, score, score_id, tmp);
    }
    _kann_evaluate_recommendation(hits, ndcgs, n_val_row, 0);
    eval_time = MPI_Wtime() - start_time;
    fprintf(stdout, "[%.2f]\n", eval_time);

    for (i = 0; i < max_epoch; ++i) {
        int n_proc = 0, n_train_err = 0, n_val_err = 0, n_train_base = 0, n_val_base = 0;
        double train_cost = 0.0, val_cost = 0.0;

        /* Training */
        kann_switch(ann, 1);
        _kann_gen_epoch(dok, max_user_id, max_item_id, n, num_neg, x, x0, y0, shuf_ext);
        start_time = MPI_Wtime();
        while (n_proc < n_train) {
            int b, c, ms = n_train - n_proc < mini_size? n_train - n_proc : mini_size;
            _kann_gen_batch(max_user_id, max_item_id, x1, y1, x0, y0, n_in, n_out, n_proc, tmp_oh, ms);
            kann_set_batch_size(ann, ms);
            train_cost += kann_cost(ann, 0, 1) * ms;
            kann_RMSprop(n_var, lr, 0, 0.9f, ann->g, ann->x, r);
            n_proc += ms;
        }
        train_cost /= n_train;
        training_time = MPI_Wtime() - start_time;

        /* Evaluate */
        kann_switch(ann, 0);
        n_proc = 0;
        for (int i = 0; i < n_val_row; i++) {
            _kann_gen_validation(i, n_val_col, _val, x1, y1, tmp_oh, n_in, max_user_id);
            kann_set_batch_size(ann, n_val_col);
            kann_cost(ann, 0, 0);
            _kann_recommender_error(ann, &hits[i], &ndcgs[i], n_val_col, top_K, score, score_id, tmp);
        }
        _kann_evaluate_recommendation(hits, ndcgs, n_val_row, train_cost);
        epoch_time = MPI_Wtime() - start_time;

        if (kann_verbose >= 3 && rank == 0) {
            fprintf(stdout, "Training: %.4f Epoch: %.4f\n", training_time, epoch_time);
        }
        if (i >= max_drop_streak && n_val_row > 0) {
            if (val_cost < min_val_cost) {
                min_set = 1;
                memcpy(min_x, ann->x, n_var * sizeof(float));
                memcpy(min_c, ann->c, n_const * sizeof(float));
                drop_streak = 0;
                min_val_cost = (float)val_cost;
            } else if (++drop_streak >= max_drop_streak)
                lr /= 10;
                //break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (min_set) {
        memcpy(ann->x, min_x, n_var * sizeof(float));
        memcpy(ann->c, min_c, n_const * sizeof(float));
    }

    free(min_c); free(min_x); free(y1); free(x1); free(y); free(x); free(shuf); free(r);
    free(shuf_ext); free(y0); free(tmp_oh); free(hits); free(ndcgs); free(score); free(score_id); 
    //float **x0    

    return i;
}

float kann_cost_fnn1(kann_t *ann, int n, float **x, float **y)
{
    int n_in, n_out, n_proc = 0, mini_size = 64 < n? 64 : n;
    float *x1, *y1;
    double cost = 0.0;

    n_in = kann_dim_in(ann);
    n_out = kann_dim_out(ann);
    if (n <= 0 || n_in < 0 || n_out < 0) return 0.0;

    x1 = (float*)malloc(n_in  * mini_size * sizeof(float));
    y1 = (float*)malloc(n_out * mini_size * sizeof(float));
    kann_feed_bind(ann, KANN_F_IN,    0, &x1);
    kann_feed_bind(ann, KANN_F_TRUTH, 0, &y1);
    kann_switch(ann, 0);
    while (n_proc < n) {
        int b, ms = n - n_proc < mini_size? n - n_proc : mini_size;
        for (b = 0; b < ms; ++b) {
            memcpy(&x1[b*n_in],  x[n_proc+b], n_in  * sizeof(float));
            memcpy(&y1[b*n_out], y[n_proc+b], n_out * sizeof(float));
        }
        kann_set_batch_size(ann, ms);
        cost += kann_cost(ann, 0, 0) * ms;
        n_proc += ms;
    }
    free(y1); free(x1);
    return (float)(cost / n);
}

const float *kann_apply1(kann_t *a, float *x)
{
    int i_out;
    i_out = kann_find(a, KANN_F_OUT, 0);
    if (i_out < 0) return 0;
    kann_set_batch_size(a, 1);
    kann_feed_bind(a, KANN_F_IN, 0, &x);
    kad_eval_at(a->n, a->v, i_out);
    return a->v[i_out]->x;
}
