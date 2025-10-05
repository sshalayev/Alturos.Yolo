using Alturos.Yolo.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;

namespace Alturos.Yolo
{
    
    public class YoloMultiWrapper : IDisposable
    {
        public const int MaxObjects = 1000;
        private const string YoloLibraryCpu = "yolo_cpp_dll_cpu.dll";
        private const string YoloLibraryGpu = "yolo_cpp_dll_gpu.dll";

        private readonly ImageAnalyzer _imageAnalyzer = new ImageAnalyzer();
        private readonly IYoloSystemValidator _yoloSystemValidator;
        private YoloObjectTypeResolver _objectTypeResolver;
        private string _name = string.Empty;

        public DetectionSystem DetectionSystem { get; private set; } = DetectionSystem.CPU;

        #region DllImport Common
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int _InitializeYolo(string configurationFilename, string weightsFilename, int gpuIndex);
        private _InitializeYolo InitializeYolo;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int _DetectImage(string filename, ref BboxContainer container);
        private _DetectImage DetectImage;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int _DetectMat(IntPtr pArray, int nSize, ref BboxContainer container);
        private _DetectMat DetectMat;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int _DisposeYolo();
        private _DisposeYolo DisposeYolo;

        #endregion

        #region DllImport Cpu

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate bool _BuiltWithOpenCV();
        private _BuiltWithOpenCV BuiltWithOpenCV;

        #endregion

        #region DllImport Gpu

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int _InitializeYoloGpuWithBatchSize(string configurationFilename, string weightsFilename, int gpuIndex, int batchSize);
        private _InitializeYoloGpuWithBatchSize InitializeYoloGpuWithBatchSize;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int _GetDeviceCount();
        private _GetDeviceCount GetDeviceCount;
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int _GetDeviceName(int gpu, StringBuilder deviceName);
        private _GetDeviceName GetDeviceName;
        

        #endregion

        
        private YoloMultiWrapper(DetectionSystem detectionSystem, string namesFile) 
        { 
            DetectionSystem = detectionSystem;
            _yoloSystemValidator = new DefaultYoloSystemValidator();
            _objectTypeResolver = new YoloObjectTypeResolver(namesFile);
            _name = Path.GetFileNameWithoutExtension(namesFile);
            var baseLibPath = DetectionSystem.GPU == detectionSystem
                    ? YoloLibraryGpu : YoloLibraryCpu;
            
            var instLibPath = Path.Combine("instances", _name, baseLibPath);
            instLibPath = Path.GetFullPath(instLibPath);
            var dir = Path.GetDirectoryName(instLibPath);
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);
            if (!File.Exists(instLibPath))
                File.Copy(baseLibPath, instLibPath);

            InitializeYolo = FunctionLoader.LoadFunction<_InitializeYolo>(instLibPath, "init");
            DetectImage = FunctionLoader.LoadFunction<_DetectImage>(instLibPath, "detect_image");
            DetectMat = FunctionLoader.LoadFunction<_DetectMat>(instLibPath, "detect_mat");
            DisposeYolo = FunctionLoader.LoadFunction<_DisposeYolo>(instLibPath, "dispose");

            if (DetectionSystem.CPU == detectionSystem)
            {
                BuiltWithOpenCV = FunctionLoader.LoadFunction<_BuiltWithOpenCV>(instLibPath, "built_with_opencv");
            }
            else if (DetectionSystem.GPU == detectionSystem) 
            {
                InitializeYoloGpuWithBatchSize = FunctionLoader.LoadFunction<_InitializeYoloGpuWithBatchSize>(instLibPath, "init");
                GetDeviceCount = FunctionLoader.LoadFunction<_GetDeviceCount>(instLibPath, "get_device_count");
                GetDeviceName = FunctionLoader.LoadFunction<_GetDeviceName>(instLibPath, "get_device_name");
            }
        }

        /// <summary>
        /// Initialize Yolo
        /// </summary>
        /// <param name="configurationFilename">Yolo configuration (.cfg) file path</param>
        /// <param name="weightsFilename">Yolo trained data (.weights) file path</param>
        /// <param name="namesFilename">Yolo object names (.names) file path</param>
        /// <param name="gpuConfig">Gpu Index if multiple graphic devices available</param>
        /// <param name="yoloSystemValidator">Yolo System validator</param>
        /// <exception cref="NotSupportedException">Thrown when the process not run in 64bit</exception>
        /// <exception cref="YoloInitializeException">Thrown if an error occurs during initialization</exception>
        public YoloMultiWrapper(string configurationFilename, string weightsFilename, string namesFilename, IYoloSystemValidator yoloSystemValidator = null) : this(DetectionSystem.CPU, namesFilename)
        {
            if (null != yoloSystemValidator)
                _yoloSystemValidator = yoloSystemValidator;
            Initialize(configurationFilename, weightsFilename, namesFilename, null);
        }


        /// <summary>
        /// Initialize Yolo
        /// </summary>
        /// <param name="configurationFilename">Yolo configuration (.cfg) file path</param>
        /// <param name="weightsFilename">Yolo trained data (.weights) file path</param>
        /// <param name="namesFilename">Yolo object names (.names) file path</param>
        /// <param name="gpuConfig">Gpu Index if multiple graphic devices available</param>
        /// <param name="yoloSystemValidator">Yolo System validator</param>
        /// <exception cref="NotSupportedException">Thrown when the process not run in 64bit</exception>
        /// <exception cref="YoloInitializeException">Thrown if an error occurs during initialization</exception>
        public YoloMultiWrapper(string configurationFilename, string weightsFilename, string namesFilename, GpuConfig gpuConfig, IYoloSystemValidator yoloSystemValidator = null) : this(DetectionSystem.GPU, namesFilename)
        {
            if (null != yoloSystemValidator)
                _yoloSystemValidator = yoloSystemValidator;
            Initialize(configurationFilename, weightsFilename, namesFilename, gpuConfig);
        }

        public void Dispose()
        {
            DisposeYolo();
        }

        private void LoadLibrary() 
        {
            var baseLibPath = DetectionSystem.GPU == DetectionSystem
                    ? YoloLibraryGpu : YoloLibraryCpu;

            var instLibPath = Path.GetFullPath(Path.Combine("instances", _name, baseLibPath));
            var dir = Path.GetDirectoryName(instLibPath);
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);
            if (!File.Exists(instLibPath))
                File.Copy(baseLibPath, instLibPath);

            InitializeYolo = FunctionLoader.LoadFunction<_InitializeYolo>(instLibPath, "init");
            DetectImage = FunctionLoader.LoadFunction<_DetectImage>(instLibPath, "detect_image");
            DetectMat = FunctionLoader.LoadFunction<_DetectMat>(instLibPath, "detect_mat");
            DisposeYolo = FunctionLoader.LoadFunction<_DisposeYolo>(instLibPath, "dispose");

            if (DetectionSystem.CPU == DetectionSystem)
            {
                BuiltWithOpenCV = FunctionLoader.LoadFunction<_BuiltWithOpenCV>(instLibPath, "built_with_opencv");
            }
            else if (DetectionSystem.GPU == DetectionSystem)
            {
                InitializeYoloGpuWithBatchSize = FunctionLoader.LoadFunction<_InitializeYoloGpuWithBatchSize>(instLibPath, "init");
                GetDeviceCount = FunctionLoader.LoadFunction<_GetDeviceCount>(instLibPath, "get_device_count");
                GetDeviceName = FunctionLoader.LoadFunction<_GetDeviceName>(instLibPath, "get_device_name");
            }
        }

        private void Initialize(string configurationFilename, string weightsFilename, string namesFilename, GpuConfig gpuConfig, int batchSize = 1)
        {
            if (IntPtr.Size != 8)
            {
                throw new NotSupportedException("Only 64-bit processes are supported");
            }

            var systemReport = _yoloSystemValidator.Validate();
            if (!systemReport.MicrosoftVisualCPlusPlusRedistributableExists)
            {
                //throw new YoloInitializeException("Microsoft Visual C++ 2017-2019 Redistributable (x64)");
            }

            int gpuIndex = 0;
            if (DetectionSystem == DetectionSystem.GPU)
            {
                if (!systemReport.CudaExists)
                {
                    throw new YoloInitializeException("CUDA files not found");
                }

                if (!systemReport.CudnnExists)
                {
                    throw new YoloInitializeException("cuDNN not found");
                }

                
                var deviceCount = null != GetDeviceCount ? GetDeviceCount() : 0;
                if (deviceCount == 0)
                {
                    throw new YoloInitializeException("No NVIDIA graphic device is available");
                }

                if (gpuConfig?.GpuIndex >= deviceCount)
                {
                    throw new YoloInitializeException("Graphic device index is out of range");
                }
                gpuIndex = gpuConfig?.GpuIndex ?? -1;
            }

            switch (DetectionSystem)
            {
                case DetectionSystem.CPU:
                    InitializeYolo(configurationFilename, weightsFilename, 0);
                    break;
                case DetectionSystem.GPU:
                    InitializeYolo(configurationFilename, weightsFilename, gpuIndex);
                    break;
            }

            
        }

        /// <summary>
        /// Detect objects on an image
        /// </summary>
        /// <param name="filepath"></param>
        /// <returns></returns>
        /// <exception cref="FileNotFoundException">Thrown when the filepath is wrong</exception>
        public IEnumerable<YoloItem> Detect(string filepath)
        {
            if (!File.Exists(filepath))
            {
                throw new FileNotFoundException("Cannot find the file", filepath);
            }

            var container = new BboxContainer();
            var count = DetectImage(filepath, ref container);

            if (count == -1)
            {
                throw new NotImplementedException("C++ dll compiled incorrectly");
            }

            return Convert(container);
        }

        /// <summary>
        /// Detect objects on an image
        /// </summary>
        /// <param name="imageData"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException">Thrown when the yolo_cpp dll is wrong compiled</exception>
        /// <exception cref="Exception">Thrown when the byte array is not a valid image</exception>
        public unsafe IEnumerable<YoloItem> Detect(byte[] imageData)
        {
            if (!_imageAnalyzer.IsValidImageFormat(imageData))
            {
                throw new Exception("Invalid image data, wrong image format");
            }

            var container = new BboxContainer();
            var count = 0;
            try
            {
                fixed (byte* pnt = imageData)
                {
                    count = DetectMat((IntPtr)pnt, imageData.Length, ref container);
                }
            }
            catch (Exception)
            {
                return null;
            }

            if (count == -1)
            {
                throw new NotImplementedException("C++ dll compiled incorrectly");
            }

            return Convert(container);
        }

        /// <summary>
        /// Detect objects on an image
        /// </summary>
        /// <param name="imagePtr"></param>
        /// <param name="size"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException">Thrown when the yolo_cpp dll is wrong compiled</exception>
        public IEnumerable<YoloItem> Detect(IntPtr imagePtr, int size)
        {
            var container = new BboxContainer();

            var count = 0;
            try
            {
                count = DetectMat(imagePtr, size, ref container);
            }
            catch (Exception)
            {
                return null;
            }

            if (count == -1)
            {
                throw new NotImplementedException("C++ dll compiled incorrectly");
            }

            return Convert(container);
        }

        public string GetGraphicDeviceName(GpuConfig gpuConfig)
        {
            if (DetectionSystem != DetectionSystem.GPU)
            {
                return CpuInfo.GetModel();
            }

            var systemReport = _yoloSystemValidator.Validate();
            if (!systemReport.CudaExists || !systemReport.CudnnExists)
            {
                return "unknown";
            }
            if (null != GetDeviceName) 
            {
                var deviceName = new StringBuilder(); //allocate memory for string
                GetDeviceName(gpuConfig.GpuIndex, deviceName);
                return deviceName.ToString();
            }
            
            return string.Empty;
        }

        public bool IsBuiltWithOpenCV()
        {
            return null != BuiltWithOpenCV ? BuiltWithOpenCV() : false;
        }

        private IEnumerable<YoloItem> Convert(BboxContainer container)
        {
            return container.candidates.Where(o => o.h > 0 || o.w > 0).Select(o =>

                new YoloItem
                {
                    X = (int)o.x,
                    Y = (int)o.y,
                    Height = (int)o.h,
                    Width = (int)o.w,
                    Confidence = o.prob,
                    Type = _objectTypeResolver.Resolve((int)o.obj_id)
                }
            );
        }
    }
}
