/*
    SPH implementation by Tommy Zhong

    Sources:
        House & Keyser 2017
        Ihmsen et al. 2014
*/

using System.Threading;
using System.Diagnostics;
using UnityEngine;

public class SPH : MonoBehaviour
{
    ParticleSystem m_sys;
    ParticleSystem.Particle[] m_particles;
    Vector3[,] m_kernelgrad_buf;
    float[,] m_kernel_buf;
    float[] m_densities;
    float[] m_pressures;
    int m_size;
    float m_dt;

    public float mass = 0.0f;
    public float rest_density = 1.0f;
    public float smooth_range = 1.0f;       // also known as h
    public float stiffness = 1.0f;          // also known as k
    public float viscosity = 1.0f;

    public int thread_count = 8;
    Thread[] m_threads;

    Stopwatch m_watch;

    void Start()
    {
        m_watch = new Stopwatch();
        m_watch.Start();
    }

    void FixedUpdate()
    {
        // check time
        m_watch.Stop();
        print(m_watch.ElapsedMilliseconds * .001);
        m_watch.Restart();

        // update attributes if necessary
        AttributeUpdate();

        // update kernel buffers and densities
        Precompute();

        // update velocities using threads
        int thread_load = m_size / thread_count;
        for (int i = 0; i < thread_count; i++)
        {
            int start = thread_load * i;
            int end = start + thread_load;
            m_threads[i] = new Thread(() => ThreadTaskAcceleration(start, end));
            m_threads[i].Start();
        }

        // perform leftover task in main thread
        ThreadTaskAcceleration(thread_load * thread_count, m_size);

        // write results to particle system
        for (int i = 0; i < thread_count; i++) m_threads[i].Join();
        m_sys.SetParticles(m_particles, m_size);
    }

    // work assigned to each thread
    void ThreadTaskAcceleration(int start, int end)
    {
        for (int i = start; i < end; i++)
        {
            m_particles[i].velocity += GetAcceleration(i) * m_dt;
        }
    }

    void AttributeUpdate()
    {
        // initialize
        if (m_sys == null)
        {
            m_sys = GetComponent<ParticleSystem>();
        }

        // set constants to positive
        rest_density = Mathf.Abs(rest_density);
        smooth_range = Mathf.Abs(smooth_range);
        stiffness = Mathf.Abs(stiffness);
        viscosity = Mathf.Abs(viscosity);

        // default particle mass
        // from Ihmsen et al. Alg. 1
        if (mass <= 0.0f)
        {
            mass = rest_density * Mathf.Pow(smooth_range, 3);
        }

        // reallocate buffers
        if (m_particles == null || m_particles.Length < m_sys.main.maxParticles)
        {
            int maxsize = m_sys.main.maxParticles;
            m_particles = new ParticleSystem.Particle[maxsize];
            m_densities = new float[maxsize];
            m_pressures = new float[maxsize];
            m_kernel_buf = new float[maxsize, maxsize];
            m_kernelgrad_buf = new Vector3[maxsize, maxsize];
            for (int i = 0; i < maxsize; i++)
            {
                m_densities[i] = rest_density;
            }
        }

        if (m_threads == null || m_threads.Length < thread_count)
        {
            m_threads = new Thread[thread_count];
        }

        m_size = m_sys.GetParticles(m_particles);
        m_dt = Time.fixedDeltaTime; // copy this value for threads
    }

    // total acceleration per update
    Vector3 GetAcceleration(int index)
    {
        Vector3 ret;

        // pressure
        ret = GetPressureAcceleration(index);

        // diffusion
        ret += GetDiffusionAcceleration(index);

        return ret;
    }

    // compute each particle's density using kernel function
    // run this before getAcceleration()
    void Precompute()
    {
        // compute kernel outputs
        // w(i,j) == w(j,i)
        // grad(w(i,j)) == -grad(w(j,i))

        // TODO: use threads
        // TODO: neighbour search optimization
        for (int i = 0; i < m_size; i++)
        {
            for (int j = 0; j < m_size; j++)
            {
                if (j > i)
                {
                    m_kernel_buf[i, j] = Kernel(i, j);
                    m_kernelgrad_buf[i, j] = KernelGrad(i, j);
                }
                else if (j < i)
                {
                    m_kernel_buf[i, j] = m_kernel_buf[j, i];
                    m_kernelgrad_buf[i, j] = -m_kernelgrad_buf[j, i];
                }
            }
        }

        // density and pressure
        for (int i = 0; i < m_size; i++)
        {
            // update densities
            // House & Keyser Eq. (14.3)
            m_densities[i] = 0.0f;
            for (int j = 0; j < m_size; j++)
            {
                if (j != i)
                {
                    m_densities[i] += m_kernel_buf[i, j];
                }
            }
            m_densities[i] *= mass;

            // update pressures (Tait with gamma = 1)
            // House & Keyser Eq. (14.5)
            
            m_pressures[i] = stiffness * (m_densities[i] - rest_density);
            // alternative equation (Tait with gamma = 7)
            // Ihmsen et al. Eq. (9)
            // stiffness * Mathf.Pow(m_densities[i] / rest_density, 7) - 1
            // not compatible with single precision float
        }
    }

    // get acceleration from pressure on one particle
    Vector3 GetPressureAcceleration(int index)
    {
        // find pressure force with mass 1
        // House & Keyser Eq. (14.6)
        Vector3 a = Vector3.zero;
        float p_i = m_pressures[index] * Mathf.Pow(m_densities[index], -2);
        for (int j = 0; j < m_size; j++)
        {
            if (j != index)
            {
                float p_j = m_pressures[j] * Mathf.Pow(m_densities[j], -2);
                a -= (p_i + p_j) * m_kernelgrad_buf[index, j];
            }
        }
        a *= mass;
        return a;
    }

    // get acceleration from diffusion/viscosity on one particle
    Vector3 GetDiffusionAcceleration(int index)
    {
        Vector3 a = Vector3.zero;
        if (viscosity > 0.0f)
        {
            for (int j = 0; j < m_size; j++)
            {
                if (index != j)
                {
                    // Ihmsen et al. Eq. (8)
                    // an approximation that avoids computing Laplacian
                    Vector3 x_ij = m_particles[index].position - m_particles[j].position;
                    Vector3 v_ij = m_particles[index].velocity - m_particles[j].velocity;
                    float f = Vector3.Dot(x_ij, m_kernelgrad_buf[index, j])
                        * (1.0f / (m_densities[j] *
                        (Vector3.Dot(x_ij, x_ij) + 0.01f * smooth_range * smooth_range)));

                    a += f * v_ij;
                }
            }
        }
        a *= viscosity * 2.0f * mass;
        return a;
    }

    // kernel function
    // Ihmsen et al. Eq. (4) & (5)
    float Kernel(int i, int j)
    {
        float length = (m_particles[i].position - m_particles[j].position)
            .magnitude * (1.0f / smooth_range);
        if (length >= 2.0f) return 0.0f;

        float output = 3.0f / (2.0f * Mathf.PI * Mathf.Pow(smooth_range, 3));
        if (length >= 0.0f && length < 1.0f)
        {
            output *= 2.0f / 3.0f - Mathf.Pow(length, 2) + 0.5f * Mathf.Pow(length, 3);
        }
        else if (length < 2.0f)
        {
            output *= 1.0f / 6.0f * Mathf.Pow(2.0f - length, 3);
        }
        return output;
    }

    // kernel function gradient
    // chain rule: grad(w(||x_i - x_j|| / h)) = w'(||x_i - x_j|| / h) * grad(||x_i - x_j|| / h)
    Vector3 KernelGrad(int i, int j)
    {
        Vector3 v = m_particles[i].position - m_particles[j].position;
        float length = v.magnitude * (1.0f / smooth_range);
        if (length >= 2.0f) return Vector3.zero;

        // derivative of Kernel()
        // incrase power of h for later
        float output = 3.0f / (2.0f * Mathf.PI * Mathf.Pow(smooth_range, 4));
        if (length >= 0.0f && length < 1.0f)
        {
            output *= -2.0f * length + 1.5f * Mathf.Pow(length, 2);
        }
        else if (length < 2.0f)
        {
            output *= 0.5f * Mathf.Pow(2.0f - length, 2);
        }

        // multiply result by gradient of v.magnitude()
        // turns out it is the same as v.normalized
        // chain rule:
        //   grad(sqrt(x^2 + y^2 + z^2) / h).x
        // = 0.5 / (h * sqrt(x^2 + y^2 + z^2)) * 2x
        // = x / (h * sqrt(x^2 + y^2 + z^2))
        return v.normalized * output;
    }
}
