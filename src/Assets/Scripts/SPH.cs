/*
    SPH implementation by Tommy Zhong

    Sources:
        House & Keyser 2017
        Ihmsen et al. 2014
*/

using System.Diagnostics;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.ParticleSystemJobs;

public class SPH : MonoBehaviour
{
    ParticleSystem m_sys;
    NativeArray<Vector3> m_positions;
    NativeArray<Vector3> m_velocities;
    NativeArray<Vector3> m_gradients;
    NativeArray<float> m_kernels;
    NativeArray<float> m_densities;
    NativeArray<float> m_pressures;
    int m_size = 0;
    float m_dt;

    public float mass = 0.0f;
    public float rest_density = 1.0f;
    public float smooth_range = 1.0f;       // also known as h
    public float stiffness = 1.0f;          // also known as k
    public float viscosity = 1.0f;

    public int thread_count = 8;

    Stopwatch m_watch;

    void OnEnable()
    {
        m_watch = new Stopwatch();
        m_watch.Start();
    }

    void OnDisable()
    {
        m_positions.Dispose();
        m_velocities.Dispose();
        m_gradients.Dispose();
        m_kernels.Dispose();
        m_densities.Dispose();
        m_pressures.Dispose();
    }

    void FixedUpdate()
    {
        // check time
        m_watch.Stop();
        print(m_watch.ElapsedMilliseconds * .001);
        m_watch.Restart();

        // update attributes if necessary
        AttributeUpdate();

        // update particle data
        var dataJob = new ParticleDataJob()
        {
            positions = m_positions,
            velocities = m_velocities
        };
        var dataHandle = dataJob.Schedule(m_sys, m_size / thread_count);

        // update kernel values and kernel gradients
        // w(i,j) == w(j,i)
        // grad(w(i,j)) == -grad(w(j,i))
        // TODO: neighbour search optimization
        var kernelJob = new KernelJob()
        {
            positions = m_positions,
            length = m_size,
            smooth_range = smooth_range,
            kernels = m_kernels
        };
        var kernelHandle = kernelJob.Schedule(m_size * m_size,
            m_size * m_size / thread_count, dataHandle);

        var kernelGradJob = new KernelGradJob()
        {
            positions = m_positions,
            length = m_size,
            smooth_range = smooth_range,
            gradients = m_gradients
        };
        var kernelGradHandle = kernelGradJob.Schedule(m_size * m_size,
            m_size * m_size / thread_count, dataHandle);

        // update densities and pressures using kernel values
        var dpJob = new DensitiesPressuresJob()
        {
            kernels = m_kernels,
            length = m_size,
            delta_time = m_dt,
            stiffness = stiffness,
            rest_density = rest_density,
            mass = mass,
            densities = m_densities,
            pressures = m_pressures
        };
        var dpHandle = dpJob.Schedule(m_size, m_size / thread_count, kernelHandle);

        // update velocities by accelerations
        var accelerationJob = new AccelerationJob()
        {
            positions = m_positions,
            velocities = m_velocities,
            gradients = m_gradients,
            densities = m_densities,
            pressures = m_pressures,
            mass = mass,
            viscosity = viscosity,
            smooth_range = smooth_range,
            delta_time = m_dt
        };

        var accelerationHandle = accelerationJob.Schedule(m_sys, m_size / thread_count,
            JobHandle.CombineDependencies(kernelGradHandle, dpHandle));
        accelerationHandle.Complete();
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

        // reallocate buffers if needed
        if (m_size < m_sys.main.maxParticles)
        {
            m_size = m_sys.main.maxParticles;

            m_positions = new NativeArray<Vector3>(m_size, Allocator.Persistent);
            m_velocities = new NativeArray<Vector3>(m_size, Allocator.Persistent);
            m_gradients = new NativeArray<Vector3>(m_size * m_size, Allocator.Persistent);
            m_kernels = new NativeArray<float>(m_size * m_size, Allocator.Persistent);
            m_densities = new NativeArray<float>(m_size, Allocator.Persistent);
            m_pressures = new NativeArray<float>(m_size, Allocator.Persistent);
            for (int i = 0; i < m_size; i++)
            {
                m_densities[i] = rest_density;
            }
        }
        m_dt = Time.fixedDeltaTime; // copy this value for jobs
    }

    // get acceleration from pressure on one particle
    static Vector3 GetPressureAcceleration(NativeArray<Vector3> gradients,
        NativeArray<float> densities, NativeArray<float> pressures, int index, float mass)
    {
        // find pressure force with mass 1
        // House & Keyser Eq. (14.6)
        Vector3 a = Vector3.zero;
        float p_i = pressures[index] * Mathf.Pow(densities[index], -2);
        int size = pressures.Length;
        for (int j = 0; j < size; j++)
        {
            if (j != index)
            {
                float p_j = pressures[j] * Mathf.Pow(densities[j], -2);
                a -= (p_i + p_j) * FindKernelGrad(gradients, index, j, size);
            }
        }
        a *= mass;
        return a;
    }

    // get acceleration from diffusion/viscosity on one particle
    static Vector3 GetDiffusionAcceleration(NativeArray<Vector3> positions,
        NativeArray<Vector3> velocities, NativeArray<Vector3> gradients,
        NativeArray<float> densities, int index, float mass, float viscosity,
        float smooth_range)
    {
        Vector3 a = Vector3.zero;
        int size = positions.Length;
        if (viscosity > 0.0f)
        {
            for (int j = 0; j < size; j++)
            {
                if (index != j)
                {
                    // Ihmsen et al. Eq. (8)
                    // an approximation that avoids computing Laplacian
                    Vector3 x_ij = positions[index] - positions[j];
                    Vector3 v_ij = velocities[index] - velocities[j];
                    float f = Vector3.Dot(x_ij, FindKernelGrad(gradients, index, j, size))
                        * (1.0f / (densities[j] *
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
    static float Kernel(Vector3 posi, Vector3 posj, float smooth_range)
    {
        float length = (posi - posj).magnitude * (1.0f / smooth_range);
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
    // chain rule:
    // grad(w(||x_i - x_j|| / h)) = w'(||x_i - x_j|| / h) * grad(||x_i - x_j|| / h)
    static Vector3 KernelGrad(Vector3 posi, Vector3 posj, float smooth_range)
    {
        Vector3 v = posi - posj;
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

    // helper function to find value from flattened 2d array
    static float FindKernel(NativeArray<float> kernels, int i, int j, int length)
    {
        if (i == j) return 0.0f;
        return i < j? kernels[i + j * length] : kernels[j + i * length];
    }

    // helper function to find value from flattened 2d array
    static Vector3 FindKernelGrad(NativeArray<Vector3> gradients, int i, int j, int length)
    {
        if (i == j) return Vector3.zero;
        return i < j ? gradients[i + j * length] : -gradients[j + i * length];
    }

    // job to load particle positions and velocities into arrays
    // IJobParticleSystemParallelFor cannot seem to run more than n loops
    // dependency: none
    struct ParticleDataJob : IJobParticleSystemParallelFor
    {
        // outputs
        public NativeArray<Vector3> positions;
        public NativeArray<Vector3> velocities;

        public void Execute(ParticleSystemJobData particles, int i)
        {
            positions[i] = particles.positions[i];
            velocities[i] = particles.velocities[i];
        }
    }

    // job to compute kernel smoother value for each pair of particles
    // dependency: ParticleDataJob
    struct KernelJob : IJobParallelFor
    {
        // inputs
        [ReadOnly]
        public NativeArray<Vector3> positions;                  // size n

        [ReadOnly]
        public int length;

        [ReadOnly]
        public float smooth_range;

        // outputs
        public NativeArray<float> kernels;                      // size n^2

        public void Execute(int i)
        {
            // find the two particles' indexes
            // i = j + k * length
            int j = i % length;
            int k = i / length;

            // assign values to 2D array (only a triangle matrix is assigned)
            // TODO: use a triangle matrix structure to save wasted space
            if (j < k)
                kernels[i] = Kernel(positions[j], positions[k], smooth_range);
            else kernels[i] = 0.0f;
        }
    }

    // job to compute kernel smoother gradient for each pair of particles
    // dependency: ParticleDataJob
    struct KernelGradJob : IJobParallelFor
    {
        // inputs
        [ReadOnly]
        public NativeArray<Vector3> positions;                  // size n
        
        [ReadOnly]
        public int length;

        [ReadOnly]
        public float smooth_range;

        // outputs
        public NativeArray<Vector3> gradients;                      // size n^2

        public void Execute(int i)
        {
            // find the two particles' indexes
            // i = j + k * length
            int j = i % length;
            int k = i / length;

            // assign values to 2D array (only a triangle matrix is assigned)
            // TODO: use a triangle matrix structure to save wasted space
            if (j < k)
                gradients[i] = KernelGrad(positions[j], positions[k], smooth_range);
            else gradients[i] = Vector3.zero;
        }
    }

    // job to update density and pressure of each particle
    // dependency: KernelJob
    struct DensitiesPressuresJob : IJobParallelFor
    {
        // inputs
        [ReadOnly]
        public NativeArray<float> kernels;                      // size n^2

        [ReadOnly]
        public int length;

        [ReadOnly]
        public float delta_time;

        [ReadOnly]
        public float stiffness;

        [ReadOnly]
        public float rest_density;

        [ReadOnly]
        public float mass;

        // outputs
        public NativeArray<float> densities;                    // size n
        public NativeArray<float> pressures;                    // size n

        public void Execute(int i)
        {
            // make sure Execute() runs n times, not n^2
            // job.Schedule(loop count, batch size)

            // update densities
            // House & Keyser Eq. (14.3)
            densities[i] = 0.0f;
            for (int j = 0; j < length; j++)
            {
                if (j != i)
                {
                    densities[i] += FindKernel(kernels, i, j, length);
                }
            }
            densities[i] *= mass;

            // update pressures (Tait with gamma = 1)
            // House & Keyser Eq. (14.5)

            pressures[i] = stiffness * (densities[i] - rest_density);
            // alternative equation (Tait with gamma = 7)
            // Ihmsen et al. Eq. (9)
            // stiffness * Mathf.Pow(m_densities[i] / rest_density, 7) - 1
            // not compatible with single precision float
        }
    }

    // job to calculate and apply acceleration of each particle
    // dependencies: DensitiesPressuresJob, KernelGradJob
    struct AccelerationJob : IJobParticleSystemParallelFor
    {
        // inputs
        [ReadOnly]
        public NativeArray<Vector3> positions;                  // size n

        [ReadOnly]
        public NativeArray<Vector3> velocities;                 // size n

        [ReadOnly]
        public NativeArray<Vector3> gradients;                  // size n^2

        [ReadOnly]
        public NativeArray<float> densities;                    // size n

        [ReadOnly]
        public NativeArray<float> pressures;                    // size n

        [ReadOnly]
        public float mass;

        [ReadOnly]
        public float viscosity;

        [ReadOnly]
        public float smooth_range;

        [ReadOnly]
        public float delta_time;

        public void Execute(ParticleSystemJobData particles, int i)
        {
            // calculate pressure and diffusion accelerations
            Vector3 acceleration =
                GetPressureAcceleration(gradients, densities, pressures, i, mass) +
                GetDiffusionAcceleration(positions, velocities, gradients, densities,
                i, mass, viscosity, smooth_range);
            acceleration *= delta_time;

            // add acceleration to particle velocity
            var x = particles.velocities.x;
            x[i] += acceleration.x;
            var y = particles.velocities.y;
            y[i] += acceleration.y;
            var z = particles.velocities.z;
            z[i] += acceleration.z;
        }
    }
}
