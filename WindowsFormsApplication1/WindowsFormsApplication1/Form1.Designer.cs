namespace WindowsFormsApplication1
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.btnLoadVideo = new System.Windows.Forms.Button();
            this.btnLoadImage = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.btnRun = new System.Windows.Forms.Button();
            this.axWindowsMediaPlayer1 = new AxWMPLib.AxWindowsMediaPlayer();
            this.btnLoadHDF5 = new System.Windows.Forms.Button();
            this.btnLoadXML = new System.Windows.Forms.Button();
            this.textBoxHDF5 = new System.Windows.Forms.TextBox();
            this.textBoxXML = new System.Windows.Forms.TextBox();
            this.textBoxImage = new System.Windows.Forms.TextBox();
            this.textBoxVideo = new System.Windows.Forms.TextBox();
            this.btnOnlyClass = new System.Windows.Forms.Button();
            this.pictureBoxLoading = new System.Windows.Forms.PictureBox();
            this.btnStop = new System.Windows.Forms.Button();
            this.radioButtonColor = new System.Windows.Forms.RadioButton();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.axWindowsMediaPlayer1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxLoading)).BeginInit();
            this.SuspendLayout();
            // 
            // btnLoadVideo
            // 
            this.btnLoadVideo.Location = new System.Drawing.Point(80, 235);
            this.btnLoadVideo.Name = "btnLoadVideo";
            this.btnLoadVideo.Size = new System.Drawing.Size(107, 48);
            this.btnLoadVideo.TabIndex = 0;
            this.btnLoadVideo.Text = "Load Video";
            this.btnLoadVideo.UseVisualStyleBackColor = true;
            this.btnLoadVideo.Click += new System.EventHandler(this.btnLoadVideo_Click);
            // 
            // btnLoadImage
            // 
            this.btnLoadImage.Location = new System.Drawing.Point(80, 181);
            this.btnLoadImage.Name = "btnLoadImage";
            this.btnLoadImage.Size = new System.Drawing.Size(107, 48);
            this.btnLoadImage.TabIndex = 1;
            this.btnLoadImage.Text = "Load Image";
            this.btnLoadImage.UseVisualStyleBackColor = true;
            this.btnLoadImage.Click += new System.EventHandler(this.btnLoadImage_Click);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Location = new System.Drawing.Point(388, 51);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(718, 528);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 2;
            this.pictureBox1.TabStop = false;
            this.pictureBox1.Click += new System.EventHandler(this.pictureBox1_Click);
            // 
            // btnRun
            // 
            this.btnRun.Location = new System.Drawing.Point(193, 331);
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(122, 58);
            this.btnRun.TabIndex = 3;
            this.btnRun.Text = "Detection and Classification";
            this.btnRun.UseVisualStyleBackColor = true;
            this.btnRun.Click += new System.EventHandler(this.btnRun_Click);
            // 
            // axWindowsMediaPlayer1
            // 
            this.axWindowsMediaPlayer1.Enabled = true;
            this.axWindowsMediaPlayer1.Location = new System.Drawing.Point(388, 51);
            this.axWindowsMediaPlayer1.Name = "axWindowsMediaPlayer1";
            this.axWindowsMediaPlayer1.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("axWindowsMediaPlayer1.OcxState")));
            this.axWindowsMediaPlayer1.Size = new System.Drawing.Size(718, 566);
            this.axWindowsMediaPlayer1.TabIndex = 4;
            // 
            // btnLoadHDF5
            // 
            this.btnLoadHDF5.Location = new System.Drawing.Point(80, 80);
            this.btnLoadHDF5.Name = "btnLoadHDF5";
            this.btnLoadHDF5.Size = new System.Drawing.Size(107, 44);
            this.btnLoadHDF5.TabIndex = 5;
            this.btnLoadHDF5.Text = "Load HDF5";
            this.btnLoadHDF5.UseVisualStyleBackColor = true;
            this.btnLoadHDF5.Click += new System.EventHandler(this.btnLoadHDF5_Click);
            // 
            // btnLoadXML
            // 
            this.btnLoadXML.Location = new System.Drawing.Point(80, 130);
            this.btnLoadXML.Name = "btnLoadXML";
            this.btnLoadXML.Size = new System.Drawing.Size(107, 45);
            this.btnLoadXML.TabIndex = 6;
            this.btnLoadXML.Text = "Load XML";
            this.btnLoadXML.UseVisualStyleBackColor = true;
            this.btnLoadXML.Click += new System.EventHandler(this.btnLoadXML_Click);
            // 
            // textBoxHDF5
            // 
            this.textBoxHDF5.Location = new System.Drawing.Point(193, 91);
            this.textBoxHDF5.Name = "textBoxHDF5";
            this.textBoxHDF5.Size = new System.Drawing.Size(122, 22);
            this.textBoxHDF5.TabIndex = 7;
            // 
            // textBoxXML
            // 
            this.textBoxXML.Location = new System.Drawing.Point(193, 141);
            this.textBoxXML.Name = "textBoxXML";
            this.textBoxXML.Size = new System.Drawing.Size(122, 22);
            this.textBoxXML.TabIndex = 8;
            // 
            // textBoxImage
            // 
            this.textBoxImage.Location = new System.Drawing.Point(193, 194);
            this.textBoxImage.Name = "textBoxImage";
            this.textBoxImage.Size = new System.Drawing.Size(122, 22);
            this.textBoxImage.TabIndex = 9;
            // 
            // textBoxVideo
            // 
            this.textBoxVideo.Location = new System.Drawing.Point(193, 248);
            this.textBoxVideo.Name = "textBoxVideo";
            this.textBoxVideo.Size = new System.Drawing.Size(122, 22);
            this.textBoxVideo.TabIndex = 10;
            // 
            // btnOnlyClass
            // 
            this.btnOnlyClass.Location = new System.Drawing.Point(80, 331);
            this.btnOnlyClass.Name = "btnOnlyClass";
            this.btnOnlyClass.Size = new System.Drawing.Size(107, 58);
            this.btnOnlyClass.TabIndex = 11;
            this.btnOnlyClass.Text = "Just Classification";
            this.btnOnlyClass.UseVisualStyleBackColor = true;
            this.btnOnlyClass.Click += new System.EventHandler(this.btnOnlyClass_Click);
            // 
            // pictureBoxLoading
            // 
            this.pictureBoxLoading.Location = new System.Drawing.Point(103, 409);
            this.pictureBoxLoading.Name = "pictureBoxLoading";
            this.pictureBoxLoading.Size = new System.Drawing.Size(180, 170);
            this.pictureBoxLoading.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBoxLoading.TabIndex = 12;
            this.pictureBoxLoading.TabStop = false;
            // 
            // btnStop
            // 
            this.btnStop.Location = new System.Drawing.Point(103, 594);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(180, 59);
            this.btnStop.TabIndex = 13;
            this.btnStop.Text = "Stop";
            this.btnStop.UseVisualStyleBackColor = true;
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // radioButtonColor
            // 
            this.radioButtonColor.AutoSize = true;
            this.radioButtonColor.Location = new System.Drawing.Point(135, 304);
            this.radioButtonColor.Name = "radioButtonColor";
            this.radioButtonColor.Size = new System.Drawing.Size(101, 21);
            this.radioButtonColor.TabIndex = 14;
            this.radioButtonColor.TabStop = true;
            this.radioButtonColor.Text = "Color Mode";
            this.radioButtonColor.UseVisualStyleBackColor = true;
            this.radioButtonColor.CheckedChanged += new System.EventHandler(this.radioButtonColor_CheckedChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1150, 748);
            this.Controls.Add(this.radioButtonColor);
            this.Controls.Add(this.btnStop);
            this.Controls.Add(this.pictureBoxLoading);
            this.Controls.Add(this.btnOnlyClass);
            this.Controls.Add(this.textBoxVideo);
            this.Controls.Add(this.textBoxImage);
            this.Controls.Add(this.textBoxXML);
            this.Controls.Add(this.textBoxHDF5);
            this.Controls.Add(this.btnLoadXML);
            this.Controls.Add(this.btnLoadHDF5);
            this.Controls.Add(this.axWindowsMediaPlayer1);
            this.Controls.Add(this.btnRun);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.btnLoadImage);
            this.Controls.Add(this.btnLoadVideo);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.axWindowsMediaPlayer1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxLoading)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnLoadVideo;
        private System.Windows.Forms.Button btnLoadImage;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button btnRun;
        private AxWMPLib.AxWindowsMediaPlayer axWindowsMediaPlayer1;
        private System.Windows.Forms.Button btnLoadHDF5;
        private System.Windows.Forms.Button btnLoadXML;
        private System.Windows.Forms.TextBox textBoxHDF5;
        private System.Windows.Forms.TextBox textBoxXML;
        private System.Windows.Forms.TextBox textBoxImage;
        private System.Windows.Forms.TextBox textBoxVideo;
        private System.Windows.Forms.Button btnOnlyClass;
        private System.Windows.Forms.PictureBox pictureBoxLoading;
        private System.Windows.Forms.Button btnStop;
        private System.Windows.Forms.RadioButton radioButtonColor;
    }
}

