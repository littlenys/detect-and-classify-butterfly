using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApplication1
{
    public partial class Form1 : Form
    {
        private static string pathHDF5;
        private static string pathXML;
        private static string pathImage;
        private static string pathVideo;
        Bitmap loading = new Bitmap(@"D:/20201/AI/loading.PNG");
        public Form1()
        {
            InitializeComponent();
        }

        private void btnLoadImage_Click(object sender, EventArgs e)
        {
            axWindowsMediaPlayer1.Visible = false;
            pictureBox1.Visible = true;
            using (OpenFileDialog ofdFile = new OpenFileDialog())
            {
                ofdFile.InitialDirectory = @"D:\20201\AI\data_test\image";

                if (ofdFile.ShowDialog() == DialogResult.OK)
                {
                    var filePath = ofdFile.FileName;
                    pathImage = filePath;
                    textBoxImage.Text = Path.GetFileName(pathImage);
                    Bitmap OriginImageClone = new Bitmap(filePath),
                        cloneImg = new Bitmap(OriginImageClone.Width, OriginImageClone.Height, OriginImageClone.PixelFormat);
                    using (Graphics g = Graphics.FromImage(cloneImg))
                    {
                        g.DrawImage(OriginImageClone, 0, 0, OriginImageClone.Width, OriginImageClone.Height);
                    }
                    //pictureBox1.Image = OriginImage;
                    pictureBox1.Image = cloneImg;
                    OriginImageClone.Dispose();
                }
                ofdFile.Dispose();
            }
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void btnLoadVideo_Click(object sender, EventArgs e)
        {
            axWindowsMediaPlayer1.Visible = true;
            pictureBox1.Visible = false;
            using (OpenFileDialog ofdFile = new OpenFileDialog())
            {
                ofdFile.InitialDirectory = @"D:\20201\AI\data_test\video";

                if (ofdFile.ShowDialog() == DialogResult.OK)
                {
                    pathVideo = ofdFile.FileName;
                    axWindowsMediaPlayer1.URL = pathVideo; // This value comes from movie selection dialog
                    textBoxVideo.Text = Path.GetFileName(pathVideo);
                }
                ofdFile.Dispose();
            }
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            if (radioButtonColor.Checked)
            {
                if ((axWindowsMediaPlayer1.Visible == false) && (pictureBox1.Visible == true))
                {
                    pictureBoxLoading.Image = loading;

                    var code = "D:/20201/AI/color_detect_02.py";
                    var hdf5 = pathHDF5;
                    var xml = pathXML;
                    var image = pathImage;
                    System.Diagnostics.Process process = new System.Diagnostics.Process();
                    System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
                    startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal;
                    startInfo.FileName = "cmd.exe";
                    startInfo.Arguments = string.Format("/c python {0} --hdf5 {1} --xml {2} --image {3}", code, pathHDF5, pathXML, pathImage); ;
                    process.StartInfo = startInfo;
                    process.Start();
                    process.WaitForExit();
                    pictureBoxLoading.Image = null;

                }

                if ((axWindowsMediaPlayer1.Visible == true) && (pictureBox1.Visible == false))
                {
                    pictureBoxLoading.Image = loading;

                    var code = "D:/20201/AI/color_detect_01.py";
                    var hdf5 = pathHDF5;
                    var xml = pathXML;
                    var video = pathVideo;
                    System.Diagnostics.Process process = new System.Diagnostics.Process();
                    System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
                    startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal;
                    startInfo.FileName = "cmd.exe";
                    startInfo.Arguments = string.Format("/c python {0} --hdf5 {1} --xml {2} --video {3}", code, pathHDF5, pathXML, pathVideo); ;
                    process.StartInfo = startInfo;
                    process.Start();
                    process.WaitForExit();
                    pictureBoxLoading.Image = null;

                }
            }
            else
            {
                if ((axWindowsMediaPlayer1.Visible == false) && (pictureBox1.Visible == true))
                {
                    pictureBoxLoading.Image = loading;

                    var code = "D:/20201/AI/detect_02.py";
                    var hdf5 = pathHDF5;
                    var xml = pathXML;
                    var image = pathImage;
                    System.Diagnostics.Process process = new System.Diagnostics.Process();
                    System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
                    startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal;
                    startInfo.FileName = "cmd.exe";
                    startInfo.Arguments = string.Format("/c python {0} --hdf5 {1} --xml {2} --image {3}", code, pathHDF5, pathXML, pathImage); ;
                    process.StartInfo = startInfo;
                    process.Start();
                    process.WaitForExit();
                    pictureBoxLoading.Image = null;

                }

                if ((axWindowsMediaPlayer1.Visible == true) && (pictureBox1.Visible == false))
                {
                    pictureBoxLoading.Image = loading;

                    var code = "D:/20201/AI/detect_01.py";
                    var hdf5 = pathHDF5;
                    var xml = pathXML;
                    var video = pathVideo;
                    System.Diagnostics.Process process = new System.Diagnostics.Process();
                    System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
                    startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal;
                    startInfo.FileName = "cmd.exe";
                    startInfo.Arguments = string.Format("/c python {0} --hdf5 {1} --xml {2} --video {3}", code, pathHDF5, pathXML, pathVideo); ;
                    process.StartInfo = startInfo;
                    process.Start();
                    process.WaitForExit();
                    pictureBoxLoading.Image = null;

                }
            }
            
        }

        private void btnLoadHDF5_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofdFile = new OpenFileDialog())
            {
                ofdFile.InitialDirectory = @"D:\20201\AI\data_test\hdf5";
                if (ofdFile.ShowDialog() == DialogResult.OK)
                {
                    pathHDF5 = ofdFile.FileName;
                    textBoxHDF5.Text = Path.GetFileName(pathHDF5);
                }
                ofdFile.Dispose();
            }
        }

        private void btnLoadXML_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofdFile = new OpenFileDialog())
            {
                ofdFile.InitialDirectory = @"D:\20201\AI\data_test\xml";
                if (ofdFile.ShowDialog() == DialogResult.OK)
                {
                    pathXML = ofdFile.FileName;
                    textBoxXML.Text = Path.GetFileName(pathXML);
                }
                ofdFile.Dispose();
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void btnOnlyClass_Click(object sender, EventArgs e)
        {
            if (radioButtonColor.Checked)
            {
                if ((axWindowsMediaPlayer1.Visible == false) && (pictureBox1.Visible == true))
                {
                    pictureBoxLoading.Image = loading;

                    var code = "D:/20201/AI/color_detect_03.py";
                    var hdf5 = pathHDF5;
                    var image = pathImage;
                    System.Diagnostics.Process process = new System.Diagnostics.Process();
                    System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
                    startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal;
                    startInfo.FileName = "cmd.exe";
                    startInfo.Arguments = string.Format("/c python {0} --hdf5 {1} --image {2}", code, pathHDF5, pathImage); ;
                    process.StartInfo = startInfo;
                    process.Start();
                    process.WaitForExit();
                    pictureBoxLoading.Image = null;


                }

            }
            else
            {
                if ((axWindowsMediaPlayer1.Visible == false) && (pictureBox1.Visible == true))
                {
                    pictureBoxLoading.Image = loading;

                    var code = "D:/20201/AI/detect_03.py";
                    var hdf5 = pathHDF5;
                    var image = pathImage;
                    System.Diagnostics.Process process = new System.Diagnostics.Process();
                    System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
                    startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal;
                    startInfo.FileName = "cmd.exe";
                    startInfo.Arguments = string.Format("/c python {0} --hdf5 {1} --image {2}", code, pathHDF5, pathImage); ;
                    process.StartInfo = startInfo;
                    process.Start();
                    process.WaitForExit();
                    pictureBoxLoading.Image = null;


                }

            }
            
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            System.Environment.Exit(1);
        }

        private void radioButtonColor_CheckedChanged(object sender, EventArgs e)
        {

        }
    }
}
