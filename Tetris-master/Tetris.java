import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.awt.event.*;
public class Tetris extends Panel implements KeyListener
{
	// 2-D array that stores the board
	private int[][] board;
	//board and arraylist storing next numbers
	private Storage nextnums = new Storage();
	//storage board and number for held piece
	private int holdnum= -1;
	private int[][] storage = new int[5][5];
	private boolean stored;
	//total points used for score
	private int totalpoints = 0;
	//Fonts used for text
	private final Font roman = new Font(Font.SANS_SERIF,Font.TYPE1_FONT, 28);
	private final Font finaltext = new Font(Font.SANS_SERIF,Font.TYPE1_FONT, 100);
	//Array of color for drawing
	Color[] colors = new Color[] {null, Color.orange, Color.blue, Color.red, Color.green,
	Color.MAGENTA, Color.YELLOW, Color.cyan, Color.gray};
	//Dimensions of window, dimension of board & size of the squares
	Dimension dim = new Dimension() ;
	private int width;
	private int height;
	private int size; 
	///Variables used to set the speed at which the pieces are falling
	private int[] levelspeeds = {40,28,20,16,13,10,8,6,5,5,5,4,4,4,3,3,3,2,2,2,2,2,2,2,2,2,1};
	 //counter used to find number of times TimerTask has been called 
	private int counter;
	//Current level of the player
	private int level;
	private int delay = 100;
	// the video memory and the off-screen memory
	BufferedImage osm  ;	// the address of the off-screen memory that holds the off–screen image
	Graphics  		osg ;	// the reference to the graphics of the off–screen memory
	Graphics g;
	//Starting point of piece
	Point startingcentre;
	//Tetris piece initialized
	Piece piece;
	//Boolean value represents whether game has ended
	boolean gameover;
	// Constructor has inputs w & h, used to take user inputs on board size
	public Tetris(int w, int h) 
	{	
		stored = false;
		gameover = false;
		counter = 0;
		level = 0;
		width = w+8;
		height = h+8;
		startingcentre = new Point(width/2, 2);
		board = new int[this.height][this.width];
		piece = new Piece(startingcentre,(int)(Math.random()*7),(int)(Math.random()*3));
		//initialize 4-square wide buffer region around the board
		for (int x = 0; x<board.length; x++)
		{
			board[x][0] = 8;
			board[x][1] = 8;
			board[x][2] = 8;
			board[x][3] = 8;
			board[x][board[0].length-1] = 8;
			board[x][board[0].length-2] = 8;
			board[x][board[0].length-3] = 8;
			board[x][board[0].length-4] = 8;
		}
		for (int x = 0; x<board[0].length; x++)
		{
			board[board.length-1][x] = 8;
			board[board.length-2][x] = 8;
			board[board.length-3][x] = 8;
			board[board.length-4][x] = 8;
		}
		//create size of squares
		size = (int)(Math.min(dim.getHeight()/height, dim.getWidth()/width));
		//Add KeyListener to take in keyboard input
		addKeyListener(this);
		//Create timer task to run
		Timer tm = new Timer();
		TimerTask motion = new TimerTask()
		{
			public void run()
			{
				if(dim!= null)
				{
					//Counter is used to count the number of frames that have passed, 
					counter ++;
					//Checking if the number of frames passed matches the current level's speed
					if(counter%levelspeeds[level] == 0)
					{
						//reset frame counter
						counter = 0;
						if(!piece.fall(board) )
						{
							/*If the piece cannot fall further, copy it into the board.
							 *Shift the stored pieces and create a new piece from next numbers
							 */
							piece.copy(board);
							int nextshape = nextnums.shift();
							piece = new Piece(startingcentre,nextshape, (int)(Math.random()*3)); 
							stored = false;
							check(board);
						}
						repaint();
						check(board);
						// If the totalpoints has passed the current level
						if(totalpoints>=(level+1)*1000)
						{
							level++;
						}
						//If the game is over
						else if(endGame())
						{
							repaint();
						}
					}
				}
			}
		}; 
		tm.scheduleAtFixedRate(motion,delay,33);
		
	}
	//Checking if a row within the board is full
	public boolean isFull(int[] a)
	{
		for (int x:a)
			if(x == 0)
				return false;
		return true;
	}
	// If the buffer zone above the board is filled, return true
	public boolean endGame()
	{
		for (int x = 0; x<4; x++)
		{	
			for (int y = 4; y<board[x].length-4; y++)
			{
				if(board[x][y] != 0)
				{
					gameover = true;
					return true;
				}
					
			}
		}
		return false;
	}
	public void check(int[][] board)
	{
		//Checks which rows are full and deleting them. Increases score depending on number of rows deleted.
		int points = 0;
		ArrayList<Integer> filled = new ArrayList<Integer>();
		for (int x = 0; x<board.length-4; x++) 
		{
			if(isFull(board[x]))
				filled.add(x);
		}
		// add coordinates of filled rows, excluding buffer zone
		points = filled.size();
		for (Integer x: filled)
		{
			//deletes the filled row and shifts the board down
			for (int y = x; y>1; y--)
			{
				board[y] = board[y-1];
			}
			//create a new row at the top
			board[0] = new int[board[1].length];
			board[0][0] = 8;
			board[0][board[0].length-1] = 8;
			
		}
		
		switch(points)// Number of rows filled
		{
		case 1:
			totalpoints += 150; break;
		case 2:
			totalpoints += 400; break;
		case 3: 
			totalpoints += 700; break;
		case 4:
			totalpoints += 1000; break;
		}
	}
	public void paint(Graphics g)
	{	
		//Initial painting of board
		dim = getSize();
		// allocate a new memory for the off–screen image
		osm = new BufferedImage( dim.width, dim.height, BufferedImage.TYPE_INT_RGB ) ;
		osg = osm.getGraphics() ; // obtain the address of the off - screen graphics
		for (int m = 4; m < board.length; m++)
		{
			for (int n = 0; n < board[0].length; n++)
			{
				g.drawRect(n*size, m*size, size, size);
			}
		}
		update(g);
	}
	public void update(Graphics g)
	{
		if(gameover)
		{
			osg.setColor(Color.WHITE);
			//clear board
			osg.fillRect(0, 0, dim.width, dim.height);
			osg.setColor(Color.black);
			osg.setFont(finaltext);
			osg.drawString("GAME OVER", getSize().width/2-350, getSize().height/2);
			osg.setFont(roman);
			osg.drawString(("Final score: " + totalpoints), 
					getSize().width/2-(("Final score: " + totalpoints).length()/2)*28, 
					getSize().height/2 + 100);
			g.drawImage(osm, 0, 0, this);
		}
		else
		{
			int size = (int)(Math.min(dim.getHeight()/height, dim.getWidth()/width));
			osg.setColor(Color.WHITE);
			//clear board
			osg.fillRect(0, 0, dim.width, dim.height);
			osg.setColor(Color.BLACK);
			//draw the main board but exclude buffer regions
			for (int m = 4; m < board.length-4; m++)
			{
				for (int n =4; n < board[0].length-4; n++)
				{
					if(board[m][n]>0)
					{
						osg.setColor(colors[board[m][n]]);
						osg.fillRect(n*size, (m+3)*size, size, size);
						osg.setColor(Color.black);
					}
					osg.drawRect(n*size, (m+3)*size, size, size);
				}
			}
			//draw the stored piece
			for (int m = 0; m<storage.length; m++)
			{
				for (int n = 0;n<storage.length; n++)
				{
					if(storage[m][n]>0)
					{
						osg.setColor(colors[storage[m][n]]);
						osg.fillRect((n+5)*size, (m+1)*size, size, size);
						osg.setColor(Color.black);
					}
					osg.drawRect((n+5)*size, (m+1)*size, size, size);
				}
			}
			// draw the next 3 pieces in the side grids
			nextnums.display(osg, size, width);
			// draw the current piece
			piece.display(osg, size);
			// draw the score
			osg.setFont(roman);
			osg.drawString(("Score: " +totalpoints), 0, size  );
			osg.drawString("HOLD", 6*size, size);
			osg.drawString("NEXT", (width+1)*size, 5*size);
			// transfer offscreen image to main panel
			g.drawImage(osm, 0, 0, this); 
		}

	}
	//clears a previous array/board
	public void clear (int[][] board)
	{
		for (int x = 0; x<board.length; x++)
			for (int y = 0; y<board.length; y++)
				board[x][y] = 0;
	}
	//stores the current piece and swaps for stored piece
	public void store ()
	{
		if(!stored)
		{
			int temp;
			if(holdnum>=0) //if there is currently a piece in storage
			{
				//swap the stored shape number and the current shape number
				temp = holdnum;
				holdnum = piece.getShape();
				//create a new piece from the stored shape number
				piece = new Piece(startingcentre,temp,0);
				stored = true;
			}
			else // if there is no piece in storage
			{//
				//store the current piece, create a new piece from the next numbers, and shift next numbers by 1
				holdnum = piece.getShape();
				int m = nextnums.shift();
				piece = new Piece(startingcentre,m,(int)(Math.random()*3));
			}
			//clear the current piece from the small board, and copy the new stored piece to the board
			Piece tempiece = new Piece(new Point(2,2),holdnum,0);
			clear(storage);
			tempiece.copy(storage);
			
			repaint();
			
		}	
	}
	public void keyPressed (KeyEvent ke)
	{
		int code = ke.getKeyCode();
		if(!gameover)
		{
			switch(code) 
			{
			//piece movement
			case KeyEvent.VK_RIGHT: piece.move(board, 1, 0); repaint(); break;
			case KeyEvent.VK_LEFT: piece.move(board, -1, 0); repaint(); break;
			case KeyEvent.VK_DOWN: piece.move(board, 0, 1); repaint(); break;
			case KeyEvent.VK_Z: piece.rotate(board,1); repaint(); break;
			case KeyEvent.VK_X: piece.rotate(board,-1); repaint(); break;
			//piece dropping
			case KeyEvent.VK_SPACE: 
				piece.drop(board); 
				int nextshape = nextnums.shift();
				stored = false;
				piece = new Piece(startingcentre,nextshape,(int)(Math.random()*4));
				repaint(); 
				break;
			case KeyEvent.VK_C: 
					store();
				break;
			}
		}
	}
	//Abstract methods
	
	public void keyReleased (KeyEvent m) {}
	public void keyTyped(KeyEvent m) {}
}
