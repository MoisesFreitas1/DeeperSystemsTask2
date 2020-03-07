package deepersystems;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;

import javax.imageio.ImageIO;

import smile.base.mlp.Layer;
import smile.base.mlp.OutputFunction;
import smile.base.rbf.RBF;
import smile.classification.MLP;
import smile.classification.RBFNetwork;

public class NeuraNetworkclassifier {
	static int IMG_WIDTH = 64;
	static int IMG_HEIGHT = 64;
	static int imageN = 1;
	
    static int NInputs;
    static double[][] inputTrain;
    static double[][] inputTest;

    static float accuracy;
    static int falsoPositivo;
    static int falsoNegativo;
    static int verdadeiroPositivo;
    static int verdadeiroNegativo;

    static int[] expectedValueTrain;
    static String[] filenames;
    
    // Folders
 	static final File dirTrain = new File("/Users/Moises/Documents/Moises/Moisés/Empregos/Deeper Systems/Teste/Task2/train"); //folder with images to train
    static final File dirTest = new File("/Users/Moises/Documents/Moises/Moisés/Empregos/Deeper Systems/Teste/Task2/test"); // folder with images to test
    static final File dirTestUpRight = new File("/Users/Moises/Documents/Moises/Moisés/Empregos/Deeper Systems/Teste/Task2/testUpRight"); // folder with images after test
    static final File fileTrain = new File("/Users/Moises/Documents/Moises/Moisés/Empregos/Deeper Systems/Teste/Task2/traintruth.csv"); //file wiht ground truth

    // array of supported extensions
    static final String[] EXTENSIONS = new String[]{
        "jpg"
    };
    
    // filter to identify images based on their extensions
    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {
        @Override
        public boolean accept(final File dir, final String name) {
            for (final String ext : EXTENSIONS) {
                if (name.endsWith("." + ext)) {
                    return (true);
                }
            }
            return (false);
        }
    };
    
	public static void main(String[] args) throws IOException {
		System.out.println("Neural Network Classifier");
		
		NInputs = IMG_WIDTH * IMG_HEIGHT; // number of inputs in each input neuron
		
		// Load images to train
		System.out.println("Load images to train");
		if (dirTrain.isDirectory()) { // make sure it's a directory
			int sizeList = dirTrain.listFiles(IMAGE_FILTER).length;
			expectedValueTrain = new int[sizeList];
            double[][] pixels = new double[sizeList][];
            int indexRow = 0; // #image's order
            for (final File f : dirTrain.listFiles(IMAGE_FILTER)) {
    			BufferedImage img = null;
                try {
                    img = ImageIO.read(f);
                    
                    BufferedReader br = new BufferedReader(new FileReader(fileTrain));
            		String st;
            		while ((st = br.readLine()) != null) {
            			String[] strV = st.split(",");
            			if(f.getName().equals(strV[0])) {
            				if(strV[1].equals("upright")) {
            					expectedValueTrain[indexRow] = 0;
            				}else if(strV[1].equals("rotated_left")) {
            					expectedValueTrain[indexRow] = 1;
            				}else if(strV[1].equals("upside_down")) {
            					expectedValueTrain[indexRow] = 2;
            				}else if(strV[1].equals("rotated_right")) {
            					expectedValueTrain[indexRow] = 3;
            				}
            			}
            		}
                    
                    pixels[indexRow] = new double[NInputs];
                    int indexCol = 0;
                    double maxPixel = 0; // to normalize
                    for (int i = 0; i < IMG_WIDTH; i++) {
                        for (int j = 0; j < IMG_HEIGHT; j++) {
                        	pixels[indexRow][indexCol] = Math.abs(img.getRGB(i, j));
                        	// to normalize
                        	if(pixels[indexRow][indexCol] > maxPixel) {
                        		maxPixel = pixels[indexRow][indexCol];
                        	}
                        	//
                        	indexCol = indexCol + 1;
                        }
                    }
                    // to normalize
                    for(int i = 0; i < indexCol; i++) {
                    	pixels[indexRow][i] = pixels[indexRow][i] / maxPixel;
                    }
                    //
                    inputTrain = pixels;
                    indexRow = indexRow + 1;
                } catch (final IOException e) {
                }
            }
		}
		// Train		
		System.out.println("Train");
		int k = 4; // number of output neurons
		int epochs = 70;
        MLP net = new MLP(NInputs, Layer.sigmoid(20), Layer.sigmoid(28), Layer.sigmoid(12), Layer.mle(k, OutputFunction.SOFTMAX)); //71,02%
        
        for (int i = 0; i < epochs; i++) {
        	for (int j = 0; j < inputTrain.length; j++) {
                net.update(inputTrain[j], expectedValueTrain[j]);
            }
        }
        
        int[] predx = new int[inputTrain.length];
        double hits = 0;
        for (int i = 0; i < inputTrain.length; i++) {
            predx[i] = net.predict(inputTrain[i]);
            if(predx[i] == expectedValueTrain[i]) {
            	hits += 1;
            }
        }
        double trainError = hits/inputTrain.length;

        System.out.format("training accuracy = %.2f%%\n", 100*trainError);

		System.out.println("Load images to test");
		if (dirTest.isDirectory()) { // make sure it's a directory
			int sizeList = dirTest.listFiles(IMAGE_FILTER).length;
			filenames = new String[sizeList];
			double[][] pixels = new double[sizeList][];
            int indexRow = 0; // #image's order
            for (final File f : dirTest.listFiles(IMAGE_FILTER)) {
                BufferedImage img = null;
                try {
                    img = ImageIO.read(f);
                    filenames[indexRow] = f.getName();
                    pixels[indexRow] = new double[NInputs];
                    int indexCol = 0;
                    double maxPixel = 0; // to normalize
                    for (int i = 0; i < IMG_WIDTH; i++) {
                        for (int j = 0; j < IMG_HEIGHT; j++) {
                        	pixels[indexRow][indexCol] = Math.abs(img.getRGB(i, j));
                        	// to normalize
                        	if(pixels[indexRow][indexCol] > maxPixel) {
                        		maxPixel = pixels[indexRow][indexCol];
                        	}
                        	//
                        	indexCol = indexCol + 1;
                        }
                    }
                    // to normalize
                    for(int i = 0; i < indexCol; i++) {
                    	pixels[indexRow][i] = pixels[indexRow][i] / maxPixel;
                    }
                    //
                    inputTest = pixels;
                    indexRow = indexRow + 1;
                } catch (final IOException e) {
                	
                }
            }
            
            System.out.println("Test");
            String predFile = "pred.csv";
            PrintWriter pw = new PrintWriter(new File(predFile));
            StringBuilder sb = new StringBuilder();
            
            int[] pred = net.predict(inputTest);            
            for(int i = 0; i < pred.length; i++) {
            	if(pred[i] == 0) {
            		if (dirTest.isDirectory()) {
            			for (final File f : dirTest.listFiles(IMAGE_FILTER)) {
            				if(f.getName().equals(filenames[i])) {
            					BufferedImage dest = copy(f);
            					ImageIO.write(dest, "jpeg", new File(dirTestUpRight + "/" + f.getName())); 
            					// save csv with preds
            					sb.append(f.getName());
            					sb.append(",");
            					sb.append("upright");
            			        sb.append('\n');
        	                }
            			}
            		}
            	}
            	if(pred[i] == 1) {
            		//rotate 90 deg
            		if (dirTest.isDirectory()) {
            			for (final File f : dirTest.listFiles(IMAGE_FILTER)) {
            				if(f.getName().equals(filenames[i])) {
            					BufferedImage dest = rotateClockwise90(ImageIO.read(f));
            					ImageIO.write(dest, "jpeg", new File(dirTestUpRight + "/" + f.getName())); 
            					// save csv with preds
            					sb.append(f.getName());
            					sb.append(",");
            					sb.append("rotated_left");
            			        sb.append('\n');
        	                }
            			}
            		}
            	}
            	if(pred[i] == 2) {
            		//rotate 180 deg
            		if (dirTest.isDirectory()) {
            			for (final File f : dirTest.listFiles(IMAGE_FILTER)) {
            				if(f.getName().equals(filenames[i])) {
            					BufferedImage dest1 = rotateClockwise90(ImageIO.read(f));
            					BufferedImage dest2 = rotateClockwise90(dest1);
            					ImageIO.write(dest2, "jpeg", new File(dirTestUpRight + "/" + f.getName()));  
            					// save csv with preds
            					sb.append(f.getName());
            					sb.append(",");
            					sb.append("upside_down");
            			        sb.append('\n');
        	                }
            			}
            		}
            	}
            	if(pred[i] == 3) {
            		//rotate 270 deg
            		if (dirTest.isDirectory()) {
            			for (final File f : dirTest.listFiles(IMAGE_FILTER)) {
            				if(f.getName().equals(filenames[i])) {
            					BufferedImage dest1 = rotateClockwise90(ImageIO.read(f));
            					BufferedImage dest2 = rotateClockwise90(dest1);
            					BufferedImage dest3 = rotateClockwise90(dest2);
            					ImageIO.write(dest3, "jpeg", new File(dirTestUpRight + "/" + f.getName()));   
            					// save csv with preds
            					sb.append(f.getName());
            					sb.append(",");
            					sb.append("rotated_right");
            			        sb.append('\n');
        	                }
            			}
            		}
            	}
            	pw.write(sb.toString());
            }
            pw.close();
		}
	}
	
	public static BufferedImage rotateClockwise90(BufferedImage src) {
	    int width = src.getWidth();
	    int height = src.getHeight();

	    BufferedImage dest = new BufferedImage(height, width, src.getType());

	    Graphics2D graphics2D = dest.createGraphics();
	    graphics2D.translate((height - width) / 2, (height - width) / 2);
	    graphics2D.rotate(Math.PI / 2, height / 2, width / 2);
	    graphics2D.drawRenderedImage(src, null);

	    return dest;
	}
	
	public static BufferedImage copy(File f) throws IOException{
		BufferedImage src = ImageIO.read(f);
		int width = src.getWidth();
	    int height = src.getHeight();
	    
	    BufferedImage dest = new BufferedImage(width, height, src.getType());
	    
	    Graphics2D graphics2D = dest.createGraphics();
	    graphics2D.drawImage(src, 0, 0, width, height, null);
	    graphics2D.dispose();
	    
	    return dest;	
	  }
}
